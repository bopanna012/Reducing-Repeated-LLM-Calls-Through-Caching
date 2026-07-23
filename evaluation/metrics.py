"""
Metrics for evaluating cache quality and cost.

Actual cost with local Ollama is $0. The cost figures here are proxy estimates
using a hosted API reference rate ($0.002 / 1K tokens) to quantify compute
demand as if the same workload ran on a paid hosted service.
Token counts use whitespace splitting; real tokenizers may differ by ~15-20%.
"""

import numpy as np
from difflib import SequenceMatcher

# Hosted API reference rate used as cost proxy (actual Ollama cost = $0)
_USD_PER_1K_TOKENS = 0.002


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def token_estimate(text: str) -> int:
    """Estimate token count as whitespace-split word count (proxy)."""
    return len(str(text).split())


# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------

def exact_match_rate(returned: list, gold: list) -> float:
    """Fraction of returned answers that exactly match the gold answer.

    Case-insensitive, whitespace-stripped comparison.
    For cached answers this should be 1.0; LLM-generated answers may differ.
    """
    if not returned:
        return 0.0
    hits = sum(
        str(r).strip().lower() == str(g).strip().lower()
        for r, g in zip(returned, gold)
    )
    return hits / len(returned)


def string_similarity(a: str, b: str) -> float:
    """Character-level similarity via SequenceMatcher, in [0, 1]."""
    return SequenceMatcher(
        None,
        str(a).lower().strip(),
        str(b).lower().strip(),
    ).ratio()


def avg_string_similarity(returned: list, gold: list) -> float:
    """Mean string similarity between returned and gold answers."""
    if not returned:
        return 0.0
    sims = [string_similarity(r, g) for r, g in zip(returned, gold)]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimated_cost_usd(total_tokens: int) -> float:
    """Proxy cost estimate using a hosted API reference rate ($0.002 / 1K tokens).

    Actual cost with local Ollama is $0. This proxy quantifies compute demand
    as if the workload ran on a paid hosted service, for fair comparison.
    """
    return round((total_tokens / 1000) * _USD_PER_1K_TOKENS, 6)


# ---------------------------------------------------------------------------
# Per-variant summary helpers
# ---------------------------------------------------------------------------

def quality_report(df) -> dict:
    """Compute quality metrics from a benchmark result DataFrame.

    Expects columns: returned_answer, gold_answer.
    """
    returned = df["returned_answer"].tolist()
    gold = df["gold_answer"].tolist()
    return {
        "exact_match_rate": round(exact_match_rate(returned, gold), 4),
        "avg_string_similarity": round(avg_string_similarity(returned, gold), 4),
    }


def cost_report(df) -> dict:
    """Compute token and cost metrics from a benchmark result DataFrame.

    Expects column: tokens_consumed (LLM tokens only; 0 for cache hits).
    """
    total_tokens = int(df["tokens_consumed"].sum())
    return {
        "total_tokens_consumed": total_tokens,
        "estimated_cost_usd": estimated_cost_usd(total_tokens),
    }
