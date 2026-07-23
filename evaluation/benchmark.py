"""
Benchmark: compare no_cache vs exact vs semantic caching.

Test set includes three query types (Q1 methodology):
  paraphrase     - minor wording changes of cached originals
  exact_duplicate - verbatim repeat of a cached original
  new_question   - genuinely new topic not in cache

CLI flags:
  --shuffle        randomise query order (tests order independence, Q4)
  --seed N         random seed for reproducibility (default 42)

Set MOCK_LLM=1 to run without a live Ollama server.
"""

import os
import sys
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embeddings.encoder import generate_embedding
from cache import exact_cache as exact_mod
from cache import semantic_cache as sem_mod
from evaluation.metrics import (
    token_estimate,
    quality_report,
    cost_report,
)

CSV_PATH = os.path.join(ROOT, "data", "university_faq_dataset.csv")
OUT = os.path.join(ROOT, "results", "reports")
GRAPHS = os.path.join(ROOT, "results", "graphs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GRAPHS, exist_ok=True)

# ---------------------------------------------------------------------------
# Optional mock LLM (set MOCK_LLM=1 to avoid needing a live Ollama server)
# ---------------------------------------------------------------------------
_USE_MOCK = os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes")

if _USE_MOCK:
    print("[benchmark] MOCK_LLM=1 - using mock responses instead of Ollama")

    def _call_llm(question: str) -> str:
        time.sleep(0.3)
        return f"Mock LLM answer for: {question[:60]}"
else:
    from llm.ollama_client import ask_llm as _ollama_ask

    def _call_llm(question: str, context: str = "") -> str:
        return _ollama_ask(question=question, context=context)


# ---------------------------------------------------------------------------
# Cache population
# ---------------------------------------------------------------------------

def build_base_cache():
    """Populate exact + semantic caches with original Q&A pairs only."""
    df = pd.read_csv(CSV_PATH)
    originals = df[df["question_type"] == "original"]
    for _, row in originals.iterrows():
        q, a = row["question"], row["answer"]
        emb = generate_embedding(q)
        exact_mod.store_exact_match(q, a)
        sem_mod.store_semantic(q, emb, a)
    return originals


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def run_variant(queries: pd.DataFrame, variant_name: str, threshold: float = 0.5):
    """Run all queries through one cache variant.

    LLM results are NOT stored back into the cache so the index stays
    identical across all three variants (fair comparison).

    Returns:
        df_out   : per-query result DataFrame
        summary  : dict of aggregate metrics
    """
    rows = []
    model_calls = 0
    exact_hits = 0
    semantic_hits = 0
    latencies_all = []
    latencies_llm = []
    total_tokens_consumed = 0  # tokens sent to / received from LLM only

    for _, row in queries.iterrows():
        q = row["question"]
        gold = row["answer"]

        t0 = time.time()
        source = "llm"
        matched_q = None
        similarity = None
        answer = None

        if variant_name == "no_cache":
            answer = _call_llm(q)
            model_calls += 1

        elif variant_name == "exact":
            cached = exact_mod.get_exact_match(q)
            if cached is not None:
                answer = cached
                source = "exact"
                exact_hits += 1
            else:
                answer = _call_llm(q)
                model_calls += 1

        elif variant_name == "semantic":
            cached = exact_mod.get_exact_match(q)
            if cached is not None:
                answer = cached
                source = "exact"
                exact_hits += 1
            else:
                emb = generate_embedding(q)
                scores, indices = sem_mod.search_semantic(emb, k=1)
                top_score = float(scores[0][0])
                idx = int(indices[0][0])
                matched_q = sem_mod.questions[idx]
                similarity = top_score
                if top_score >= threshold:
                    answer = sem_mod.answers[idx]
                    source = "semantic"
                    semantic_hits += 1
                else:
                    answer = _call_llm(q)
                    model_calls += 1

        elapsed = time.time() - t0
        latencies_all.append(elapsed)
        if source == "llm":
            latencies_llm.append(elapsed)
            # Count tokens only for actual LLM calls (proxy: input + output words)
            total_tokens_consumed += token_estimate(q) + token_estimate(answer)

        rows.append({
            "query": q,
            "gold_answer": gold,
            "returned_answer": answer,
            "source": source,
            "matched_question": matched_q,
            "similarity": similarity,
            "latency": elapsed,
            "tokens_consumed": (
                token_estimate(q) + token_estimate(answer) if source == "llm" else 0
            ),
        })

    df_out = pd.DataFrame(rows)

    quality = quality_report(df_out)
    cost = cost_report(df_out)

    n = len(df_out)
    summary = {
        "variant": variant_name,
        "queries": n,
        # --- RQ1 ---
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "llm_calls": model_calls,
        "cache_hit_rate": round((exact_hits + semantic_hits) / n, 4),
        # --- RQ2 ---
        "avg_latency_all": round(pd.Series(latencies_all).mean(), 4),
        "avg_latency_llm": round(
            pd.Series(latencies_llm).mean() if latencies_llm else 0.0, 4
        ),
        "total_tokens_consumed": cost["total_tokens_consumed"],
        "estimated_cost_usd": cost["estimated_cost_usd"],
        # --- RQ3 ---
        "exact_match_rate": quality["exact_match_rate"],
        "avg_string_similarity": quality["avg_string_similarity"],
    }
    return df_out, summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_COLORS = {"no_cache": "#e15759", "exact": "#f28e2b", "semantic": "#4e79a7"}


def _variant_colors(variants):
    return [_COLORS.get(v, "gray") for v in variants]


def plot_model_calls(df_sum):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_sum["variant"], df_sum["llm_calls"],
           color=_variant_colors(df_sum["variant"]))
    ax.set_ylabel("LLM model calls")
    ax.set_title("LLM calls per variant  (RQ1 / RQ2)")
    ax.grid(axis="y", alpha=0.4)
    for i, v in enumerate(df_sum["llm_calls"]):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    out = os.path.join(GRAPHS, "benchmark_model_calls.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_hit_breakdown(df_sum):
    """Stacked bar: exact hits / semantic hits / LLM calls (RQ1)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    variants = df_sum["variant"].tolist()
    exact_h = df_sum["exact_hits"].tolist()
    sem_h = df_sum["semantic_hits"].tolist()
    llm_h = df_sum["llm_calls"].tolist()

    x = range(len(variants))
    ax.bar(x, exact_h, label="Exact cache hit", color="#59a14f")
    ax.bar(x, sem_h, bottom=exact_h, label="Semantic cache hit", color="#4e79a7")
    ax.bar(x, llm_h,
           bottom=[e + s for e, s in zip(exact_h, sem_h)],
           label="LLM call", color="#e15759")

    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.set_ylabel("Number of queries")
    ax.set_title("Request source breakdown  (RQ1)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(GRAPHS, "benchmark_hit_breakdown.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_latency(df_sum):
    """Side-by-side bars: overall avg latency vs LLM-only avg latency (RQ2)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    variants = df_sum["variant"].tolist()
    x = range(len(variants))
    width = 0.35
    ax.bar([i - width / 2 for i in x],
           df_sum["avg_latency_all"], width,
           label="Avg latency (all queries)", color="#76b7b2")
    ax.bar([i + width / 2 for i in x],
           df_sum["avg_latency_llm"], width,
           label="Avg latency (LLM calls only)", color="#b07aa1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Latency comparison  (RQ2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(GRAPHS, "benchmark_latency.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_quality(df_sum):
    """Side-by-side bars: exact match rate and string similarity (RQ3)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    variants = df_sum["variant"].tolist()
    x = range(len(variants))
    width = 0.35
    ax.bar([i - width / 2 for i in x],
           df_sum["exact_match_rate"], width,
           label="Exact match rate (vs gold)", color="#f1ce63")
    ax.bar([i + width / 2 for i in x],
           df_sum["avg_string_similarity"], width,
           label="Avg string similarity", color="#ff9d9a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Answer quality vs gold standard  (RQ3)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(GRAPHS, "benchmark_quality.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomise query order (Q4 order-control test)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle (default 42)")
    args = parser.parse_args()

    print("Building base cache from original Q&A pairs...")
    build_base_cache()

    df_all = pd.read_csv(CSV_PATH)

    # Build unified test set: paraphrase + exact_duplicate + new_question (Q1)
    test_types = ["paraphrase", "exact_duplicate", "new_question"]
    test_df = df_all[df_all["question_type"].isin(test_types)].reset_index(drop=True)

    if args.shuffle:
        test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"Query order: SHUFFLED (seed={args.seed})")
    else:
        print("Query order: FIXED (original CSV order)")

    counts = test_df["question_type"].value_counts().to_dict()
    print(f"Test set: {len(test_df)} queries  "
          f"({counts.get('paraphrase',0)} paraphrase, "
          f"{counts.get('exact_duplicate',0)} exact_duplicate, "
          f"{counts.get('new_question',0)} new_question)\n")

    summaries = []
    for variant in ["no_cache", "exact", "semantic"]:
        print(f"Running variant: {variant} ...")
        df_out, summary = run_variant(test_df, variant, threshold=0.5)
        suffix = "_shuffled" if args.shuffle else ""
        out_csv = os.path.join(OUT, f"benchmark_{variant}{suffix}.csv")
        df_out.to_csv(out_csv, index=False)
        print(f"  Saved {out_csv}")
        print(f"  hit_rate={summary['cache_hit_rate']:.1%}  "
              f"llm_calls={summary['llm_calls']}  "
              f"avg_latency={summary['avg_latency_all']:.3f}s  "
              f"cost=${summary['estimated_cost_usd']:.4f}  "
              f"exact_match={summary['exact_match_rate']:.1%}\n")
        summaries.append(summary)

    df_sum = pd.DataFrame(summaries)
    suffix = "_shuffled" if args.shuffle else ""
    sum_csv = os.path.join(OUT, f"benchmark_summary{suffix}.csv")
    df_sum.to_csv(sum_csv, index=False)
    print(f"Saved summary to {sum_csv}\n")

    if not args.shuffle:
        print("Generating plots...")
        plot_model_calls(df_sum)
        plot_hit_breakdown(df_sum)
        plot_latency(df_sum)
        plot_quality(df_sum)
    print("\nDone. All outputs in results/")


if __name__ == "__main__":
    main()
