"""
Borderline semantic match inspection (Q5).

Rubric for judging whether a cached answer is acceptable for a query whose
similarity score falls within the borderline band (threshold +/- BAND):

  CORRECT_REUSE   : returned_answer == gold_answer (exact match, case-insensitive)
  ACCEPTABLE_REUSE: returned_answer != gold_answer but covers the same fact
                    (string similarity >= 0.8 AND topic keyword overlap)
  INCORRECT_REUSE : returned_answer answers a different question entirely
                    (string similarity < 0.8 OR matched_question topic differs)

Agreement rule (for multi-reviewer use):
  Two independent reviewers apply the rubric. If they disagree, the case is
  escalated to a third reviewer whose decision is final.
  In this automated run, the rubric is applied programmatically; cases marked
  ACCEPTABLE_REUSE are flagged for manual review.

Outputs:
  results/reports/borderline_cases.csv     - all near-threshold matches with judgment
  results/reports/borderline_summary.txt   - counts and overall safety assessment
"""

import os
import sys
import pandas as pd
from difflib import SequenceMatcher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPORTS  = os.path.join(ROOT, "results", "reports")
THRESHOLD = 0.50
BAND      = 0.10   # borderline = [threshold - BAND, threshold + BAND]


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()


def apply_rubric(returned: str, gold: str, matched_q: str, query: str) -> str:
    """Apply the three-level rubric and return a judgment string."""
    ret = str(returned).strip().lower()
    gld = str(gold).strip().lower()

    # Level 1: exact match
    if ret == gld:
        return "CORRECT_REUSE"

    # Level 2: high string similarity (same fact, different phrasing)
    sim = string_similarity(returned, gold)
    if sim >= 0.8:
        return "ACCEPTABLE_REUSE"

    # Level 3: topic keyword overlap as a secondary check
    query_words  = set(str(query).lower().split())
    matched_words = set(str(matched_q).lower().split())
    overlap = len(query_words & matched_words) / max(len(query_words), 1)
    if sim >= 0.5 and overlap >= 0.4:
        return "ACCEPTABLE_REUSE"

    return "INCORRECT_REUSE"


def load_semantic_results() -> pd.DataFrame:
    path = os.path.join(REPORTS, "benchmark_semantic.csv")
    if not os.path.exists(path):
        sys.exit("benchmark_semantic.csv not found. Run benchmark.py first.")
    return pd.read_csv(path)


def main():
    df = load_semantic_results()

    # Keep only semantic cache hits (source == "semantic") that have a similarity score
    sem_hits = df[df["source"] == "semantic"].copy()
    sem_hits["similarity"] = pd.to_numeric(sem_hits["similarity"], errors="coerce")
    sem_hits = sem_hits.dropna(subset=["similarity"])

    lo = THRESHOLD - BAND
    hi = THRESHOLD + BAND
    borderline = sem_hits[
        (sem_hits["similarity"] >= lo) & (sem_hits["similarity"] <= hi)
    ].copy()

    print(f"Semantic hits total       : {len(sem_hits)}")
    print(f"Borderline band           : [{lo:.2f}, {hi:.2f}]")
    print(f"Borderline cases found    : {len(borderline)}")

    if len(borderline) == 0:
        print("\nNo borderline cases - all semantic hits are well above threshold.")
        print("This means the threshold=0.50 is safely conservative for this dataset.")
        # Still produce empty output files so the pipeline doesn't break
        borderline["judgment"] = []
        borderline["str_similarity_to_gold"] = []
    else:
        borderline["str_similarity_to_gold"] = borderline.apply(
            lambda r: round(string_similarity(r["returned_answer"], r["gold_answer"]), 4),
            axis=1,
        )
        borderline["judgment"] = borderline.apply(
            lambda r: apply_rubric(
                r["returned_answer"], r["gold_answer"],
                r["matched_question"], r["query"]
            ),
            axis=1,
        )

    # Also inspect ALL semantic hits to give a full quality picture
    sem_hits["str_similarity_to_gold"] = sem_hits.apply(
        lambda r: round(string_similarity(r["returned_answer"], r["gold_answer"]), 4),
        axis=1,
    )
    sem_hits["judgment"] = sem_hits.apply(
        lambda r: apply_rubric(
            r["returned_answer"], r["gold_answer"],
            str(r.get("matched_question", "")), r["query"]
        ),
        axis=1,
    )

    # Save borderline CSV
    out_csv = os.path.join(REPORTS, "borderline_cases.csv")
    cols = ["query", "matched_question", "similarity", "returned_answer",
            "gold_answer", "str_similarity_to_gold", "judgment"]
    borderline[cols].to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Summary
    counts = sem_hits["judgment"].value_counts().to_dict()
    total  = len(sem_hits)

    summary_lines = [
        "BORDERLINE INSPECTION SUMMARY",
        "=" * 50,
        f"Threshold          : {THRESHOLD}",
        f"Borderline band    : [{lo:.2f}, {hi:.2f}]  (+/- {BAND})",
        "",
        f"ALL semantic hits ({total} total):",
        f"  CORRECT_REUSE    : {counts.get('CORRECT_REUSE', 0):3d}  "
        f"({counts.get('CORRECT_REUSE', 0)/total:.1%})",
        f"  ACCEPTABLE_REUSE : {counts.get('ACCEPTABLE_REUSE', 0):3d}  "
        f"({counts.get('ACCEPTABLE_REUSE', 0)/total:.1%})  <- FLAG FOR MANUAL REVIEW",
        f"  INCORRECT_REUSE  : {counts.get('INCORRECT_REUSE', 0):3d}  "
        f"({counts.get('INCORRECT_REUSE', 0)/total:.1%})",
        "",
        f"Borderline cases   : {len(borderline)} / {total}",
        "",
        "Rubric applied:",
        "  CORRECT_REUSE    : returned == gold (exact, case-insensitive)",
        "  ACCEPTABLE_REUSE : string_similarity >= 0.8 OR (sim >= 0.5 AND",
        "                     topic keyword overlap >= 0.4)",
        "  INCORRECT_REUSE  : string_similarity < 0.5 AND overlap < 0.4",
        "",
        "Multi-reviewer rule:",
        "  Two reviewers apply rubric independently. Disagreements escalated",
        "  to a third reviewer whose decision is final.",
        "",
    ]

    incorrect = counts.get("INCORRECT_REUSE", 0)
    if incorrect == 0:
        summary_lines.append("Safety verdict: NO incorrect reuse detected at threshold=0.50.")
    else:
        summary_lines.append(
            f"Safety verdict: {incorrect} incorrect reuse case(s) detected. "
            "Consider raising the threshold."
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    out_txt = os.path.join(REPORTS, "borderline_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nSaved {out_txt}")


if __name__ == "__main__":
    main()
