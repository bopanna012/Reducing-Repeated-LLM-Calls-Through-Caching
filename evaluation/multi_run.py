"""
Multi-run statistical analysis (Q3 + Q4).

Q3 — Variance across runs:
  Each benchmark CSV already contains N per-query latency measurements.
  This script treats those as N independent samples and computes:
    mean, std, 95% CI (t-distribution, df = N-1)
  It also re-runs the semantic variant 5 times (fast, no LLM) to confirm
  stability across independent executions.

Q4 — Order independence:
  Runs the full benchmark once with fixed order and once with shuffled order,
  then compares hit rates and latencies to confirm query order does not affect
  cache performance (since the cache is pre-built and not updated during eval).

Outputs:
  results/reports/stats_per_variant.csv   - mean/std/CI per variant
  results/reports/order_comparison.csv    - fixed vs shuffled comparison
  results/reports/semantic_stability.csv  - 5 repeated semantic runs
  results/graphs/latency_ci.png           - CI error-bar plot
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embeddings.encoder import generate_embedding
from cache import exact_cache as exact_mod
from cache import semantic_cache as sem_mod

REPORTS = os.path.join(ROOT, "results", "reports")
GRAPHS  = os.path.join(ROOT, "results", "graphs")
CSV_PATH = os.path.join(ROOT, "data", "university_faq_dataset.csv")
THRESHOLD = 0.5
VARIANTS  = ["no_cache", "exact", "semantic"]


# ---------------------------------------------------------------------------
# Q3 — Statistics from existing per-query CSVs
# ---------------------------------------------------------------------------

def compute_stats_from_csv(variant: str) -> dict:
    """Load per-query CSV and compute mean, std, 95% CI for latency."""
    path = os.path.join(REPORTS, f"benchmark_{variant}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found - run benchmark.py first")
        return {}

    df = pd.read_csv(path)
    latencies = df["latency"].dropna().values
    n = len(latencies)
    mean = np.mean(latencies)
    std  = np.std(latencies, ddof=1)
    # 95% CI using t-distribution
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=stats.sem(latencies))

    return {
        "variant": variant,
        "n_queries": n,
        "mean_latency": round(mean, 4),
        "std_latency":  round(std, 4),
        "ci95_lo":      round(ci_lo, 4),
        "ci95_hi":      round(ci_hi, 4),
        "llm_calls":    int((df["source"] == "llm").sum()),
        "cache_hits":   int((df["source"] != "llm").sum()),
    }


def plot_ci(df_stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df_stats))
    means  = df_stats["mean_latency"].values
    ci_lo  = means - df_stats["ci95_lo"].values
    ci_hi  = df_stats["ci95_hi"].values - means
    colors = ["#e15759", "#f28e2b", "#4e79a7"]

    ax.bar(x, means, color=colors, alpha=0.8, label="Mean latency")
    ax.errorbar(x, means, yerr=[ci_lo, ci_hi], fmt="none",
                color="black", capsize=6, linewidth=1.5, label="95% CI")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_stats["variant"])
    ax.set_ylabel("Latency (s)")
    ax.set_title("Mean latency with 95% CI per variant  (Q3)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(GRAPHS, "latency_ci.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Q3 — Semantic variant repeated 5 times (stability check)
# ---------------------------------------------------------------------------

def build_cache():
    df = pd.read_csv(CSV_PATH)
    for _, row in df[df["question_type"] == "original"].iterrows():
        emb = generate_embedding(row["question"])
        exact_mod.store_exact_match(row["question"], row["answer"])
        sem_mod.store_semantic(row["question"], emb, row["answer"])


def run_semantic_once(queries: pd.DataFrame, run_id: int) -> dict:
    latencies = []
    hits = 0
    for _, row in queries.iterrows():
        q = row["question"]
        t0 = time.time()
        cached = exact_mod.get_exact_match(q)
        if cached is not None:
            hits += 1
        else:
            emb = generate_embedding(q)
            scores, indices = sem_mod.search_semantic(emb, k=1)
            if float(scores[0][0]) >= THRESHOLD:
                hits += 1
        latencies.append(time.time() - t0)

    return {
        "run": run_id,
        "n_queries": len(queries),
        "hits": hits,
        "hit_rate": round(hits / len(queries), 4),
        "mean_latency": round(np.mean(latencies), 4),
        "std_latency":  round(np.std(latencies, ddof=1), 4),
    }


# ---------------------------------------------------------------------------
# Q4 — Order independence check
# ---------------------------------------------------------------------------

def compare_order(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run semantic variant with fixed and shuffled order; compare results."""
    rows = []
    for label, df in [("fixed", test_df),
                      ("shuffled_42", test_df.sample(frac=1, random_state=42)),
                      ("shuffled_99", test_df.sample(frac=1, random_state=99))]:
        r = run_semantic_once(df.reset_index(drop=True), run_id=label)
        r["order"] = label
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Q3 part 1: stats from existing CSVs ---
    print("=== Q3: Per-variant latency statistics ===")
    stat_rows = []
    for v in VARIANTS:
        s = compute_stats_from_csv(v)
        if s:
            stat_rows.append(s)
            print(f"  {v:12s}  mean={s['mean_latency']:.4f}s  "
                  f"std={s['std_latency']:.4f}s  "
                  f"95% CI=[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]  "
                  f"n={s['n_queries']}")

    if stat_rows:
        df_stats = pd.DataFrame(stat_rows)
        out = os.path.join(REPORTS, "stats_per_variant.csv")
        df_stats.to_csv(out, index=False)
        print(f"  Saved {out}")
        plot_ci(df_stats)

    # --- Q3 part 2: semantic variant repeated 5 times ---
    print("\n=== Q3: Semantic variant stability (5 runs) ===")
    build_cache()
    df_all = pd.read_csv(CSV_PATH)
    test_types = ["paraphrase", "exact_duplicate", "new_question"]
    test_df = df_all[df_all["question_type"].isin(test_types)].reset_index(drop=True)

    stability_rows = []
    for i in range(1, 6):
        r = run_semantic_once(test_df, run_id=i)
        stability_rows.append(r)
        print(f"  Run {i}: hit_rate={r['hit_rate']:.1%}  "
              f"mean_latency={r['mean_latency']:.4f}s  "
              f"std={r['std_latency']:.4f}s")

    df_stab = pd.DataFrame(stability_rows)
    out = os.path.join(REPORTS, "semantic_stability.csv")
    df_stab.to_csv(out, index=False)
    print(f"  Saved {out}")
    print(f"  Hit rate across 5 runs: "
          f"{df_stab['hit_rate'].mean():.1%} +/- {df_stab['hit_rate'].std():.4f}")

    # --- Q4: order independence ---
    print("\n=== Q4: Order independence (fixed vs shuffled) ===")
    df_order = compare_order(test_df)
    for _, r in df_order.iterrows():
        print(f"  {r['order']:15s}  hit_rate={r['hit_rate']:.1%}  "
              f"mean_latency={r['mean_latency']:.4f}s")
    out = os.path.join(REPORTS, "order_comparison.csv")
    df_order.to_csv(out, index=False)
    print(f"  Saved {out}")

    hit_rates = df_order["hit_rate"].unique()
    if len(hit_rates) == 1:
        print(f"\n  Result: hit rate is IDENTICAL ({hit_rates[0]:.1%}) across all "
              f"orderings - query order does not affect cache performance.")
    else:
        print("\n  Result: hit rate varies across orderings - investigate further.")

    print("\nDone.")


if __name__ == "__main__":
    main()
