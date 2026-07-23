import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embeddings.encoder import generate_embedding
from cache.semantic_cache import store_semantic, search_semantic, answers

CSV = os.path.join(ROOT, "data", "university_faq_dataset.csv")
OUT_DIR = os.path.join(ROOT, "results")
REPORTS = os.path.join(OUT_DIR, "reports")
GRAPHS = os.path.join(OUT_DIR, "graphs")

os.makedirs(REPORTS, exist_ok=True)
os.makedirs(GRAPHS, exist_ok=True)


def build_index():
    df = pd.read_csv(CSV)
    originals = df[df["question_type"] == "original"]

    for _, row in originals.iterrows():
        q = row["question"]
        a = row["answer"]
        emb = generate_embedding(q)
        store_semantic(q, emb, a)

    return originals


def load_paraphrases():
    df = pd.read_csv(CSV)
    paras = df[df["question_type"] == "paraphrase"].copy()
    return paras


def evaluate_thresholds(paras, thresholds):
    y_true = []
    scores = []

    # For each paraphrase, get top match and record if matched answer equals paraphrase answer
    for _, row in paras.iterrows():
        q = row["question"]
        true_ans = row["answer"]
        emb = generate_embedding(q)
        s, idx = search_semantic(emb, k=1)
        top_score = float(s[0][0])
        matched_idx = int(idx[0][0])
        matched_ans = answers[matched_idx]

        scores.append(top_score)
        y_true.append(1 if matched_ans == true_ans else 0)

    scores = np.array(scores)
    y_true = np.array(y_true)

    results = []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        if preds.sum() == 0:
            precision = 1.0 if preds.sum() == 0 and y_true.sum() == 0 else 0.0
        else:
            precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        results.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1, "hits": int(preds.sum())})

    return pd.DataFrame(results), scores, y_true


if __name__ == "__main__":
    print("Building semantic index from original questions...")
    build_index()

    paras = load_paraphrases()
    print(f"Loaded {len(paras)} paraphrase queries for evaluation.")

    thresholds = np.linspace(0.5, 0.95, 46)

    df_results, scores, y_true = evaluate_thresholds(paras, thresholds)

    report_csv = os.path.join(REPORTS, "threshold_sweep.csv")
    df_results.to_csv(report_csv, index=False)
    print(f"Wrote threshold sweep CSV to {report_csv}")

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_results["threshold"], df_results["precision"], label="precision")
    plt.plot(df_results["threshold"], df_results["recall"], label="recall")
    plt.plot(df_results["threshold"], df_results["f1"], label="f1")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("Threshold sweep")
    plt.legend()
    plt.grid(True)
    out_png = os.path.join(GRAPHS, "threshold_sweep.png")
    plt.savefig(out_png)
    print(f"Wrote plot to {out_png}")

    # recommend threshold by max F1
    best = df_results.loc[df_results["f1"].idxmax()]
    print("Best threshold by F1:")
    print(best.to_dict())

    # Save summary
    with open(os.path.join(REPORTS, "threshold_summary.txt"), "w") as f:
        f.write(str(best.to_dict()))

    print("Done.")
