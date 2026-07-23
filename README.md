# Reducing Repeated LLM Calls Through Caching

A semantic caching system that minimizes repeated Large Language Model (LLM) calls by combining **Exact Match Caching**, **Semantic Caching using Sentence Embeddings**, and **FAISS Vector Search**.

This project demonstrates how semantic caching can significantly reduce latency, computational cost, and repeated LLM inference while maintaining high answer quality.

---

## 📖 Project Overview

LLM-based applications often send every user query to the language model, even when identical or semantically similar questions have already been answered.

This project introduces a hybrid caching approach consisting of:

- ✅ Exact Match Cache
- ✅ Semantic Cache (FAISS + Sentence Transformers)
- ✅ LLM Fallback (Ollama)

The system first checks the cache before calling the LLM, reducing unnecessary inference.

---

## 🏗️ System Architecture

```text
                User Query
                     │
                     ▼
            Exact Match Cache
             │            │
        Cache Hit     Cache Miss
             │            ▼
             │     Generate Embedding
             │            │
             │            ▼
             │      FAISS Similarity Search
             │            │
        Similarity > Threshold?
             │            │
           Yes            No
             │            ▼
      Semantic Cache     LLM (Ollama)
             │            │
             └──────► Return Answer
```

---

# ✨ Features

- Exact string matching
- Semantic similarity search using embeddings
- FAISS vector indexing
- Local LLM inference using Ollama
- Automatic cache updates
- Benchmarking framework
- Threshold tuning
- Latency and cost analysis
- Answer quality evaluation

---

# 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| LLM | Ollama (Llama 3) |
| Libraries | Sentence Transformers, NumPy, Pandas, Matplotlib |
| Evaluation | Precision, Recall, F1 Score |

---

# 📂 Project Structure

```text
Reducing-Repeated-LLM-Calls-Through-Caching/
│
├── cache/
│   ├── exact_cache.py           # Normalized-string exact-match cache
│   └── semantic_cache.py        # FAISS-backed semantic cache
│
├── embeddings/
│   └── encoder.py                # Sentence-embedding generation (all-MiniLM-L6-v2)
│
├── llm/
│   └── ollama_client.py          # Local LLM fallback (Ollama)
│
├── evaluation/
│   ├── benchmark.py              # No-cache vs exact vs semantic benchmark
│   ├── metrics.py                # Cost, similarity, and token-estimate helpers
│   ├── multi_run.py               # Statistical variance & order-independence checks
│   ├── threshold_tests.py         # Similarity-threshold sweep (precision/recall/F1)
│   ├── borderline_inspection.py   # Manual-review rubric for near-threshold matches
│   └── rq_analysis.py             # Aggregates benchmark CSVs into the RQ report/figure
│
├── data/
│   └── university_faq_dataset.csv
│
├── results/
│   ├── graphs/                   # Generated PNG figures
│   └── reports/                  # Generated CSV/TXT reports
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed locally, with the `llama3` model pulled (`ollama pull llama3`)

## Clone the repository

```bash
git clone https://github.com/Bopanna012/Reducing-Repeated-LLM-Calls-Through-Caching.git

cd Reducing-Repeated-LLM-Calls-Through-Caching
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Start Ollama

```bash
ollama run llama3
```

---

# ▶️ Run the Project

```bash
python main.py
```

Example

```text
Ask Question:

When is the application deadline?
```

Output

```text
Semantic Cache Hit

Similarity: 0.93

Answer:
Applications close on July 20.
```

---

# 📊 Benchmark Evaluation

Run

```bash
python evaluation/benchmark.py
```

The benchmark compares three approaches — **No Cache**, **Exact Cache**, and **Semantic Cache** — across paraphrase, exact-duplicate, and new-question query types.

Optional flags:

```bash
python evaluation/benchmark.py --shuffle --seed 42   # randomize query order (reproducible)
```

Set `MOCK_LLM=1` as an environment variable to run the benchmark without a live Ollama server.

Metrics evaluated:

- Cache Hit Rate
- LLM Calls
- Average Latency
- Estimated Cost
- Exact Match Rate
- String Similarity
- Precision
- Recall
- F1 Score

## Additional Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `evaluation/threshold_tests.py` | Sweeps the similarity threshold and reports precision/recall/F1 to find the best cutoff |
| `evaluation/multi_run.py` | Checks run-to-run variance (confidence intervals) and order independence of the cache |
| `evaluation/borderline_inspection.py` | Applies a manual-review rubric to near-threshold semantic matches |
| `evaluation/rq_analysis.py` | Aggregates all benchmark CSVs into a single research-question report and figure |

Run any of them the same way, e.g.:

```bash
python evaluation/threshold_tests.py
python evaluation/multi_run.py
python evaluation/borderline_inspection.py
python evaluation/rq_analysis.py   # run after benchmark.py — reads its CSV output
```

---

# 📈 Experimental Results

| Metric | No Cache | Exact Cache | Semantic Cache |
|---------|----------|-------------|----------------|
| Cache Hit Rate | 0% | 5.3% | **90.5%** |
| LLM Calls | 95 | 90 | **9** |
| Average Latency | 5.89 s | 5.14 s | **0.54 s** |
| Estimated Cost | $0.0444 | $0.0435 | **$0.0038** |
| Exact Match Rate | 0% | 5.3% | **89.5%** |

### Key Improvements

- 🚀 91% fewer LLM calls
- ⚡ 11× faster responses
- 💰 91% estimated cost reduction
- ✅ 89.5% answer quality

---

# 🔬 Methodology

1. Load FAQ dataset.
2. Build Exact Cache.
3. Generate sentence embeddings.
4. Store embeddings in FAISS.
5. Receive user query.
6. Check Exact Cache.
7. If not found, search Semantic Cache.
8. If similarity exceeds threshold (0.50), reuse cached answer.
9. Otherwise, call the LLM.
10. Return the response.

---

# 📚 Research Questions

- **RQ1** — What proportion of requests can be served from cache instead of the LLM? (`evaluation/benchmark.py`, `evaluation/rq_analysis.py`)
- **RQ2** — How much do caching strategies reduce model calls, latency, and cost? (`evaluation/benchmark.py`, `evaluation/rq_analysis.py`)
- **RQ3** — How does semantic caching compare with exact-match for answer quality and reuse? (`evaluation/rq_analysis.py`)
- **Q4** — Does query order affect cache performance? (`evaluation/multi_run.py`)
- **Q5** — Are borderline (near-threshold) semantic matches actually correct reuses? (`evaluation/borderline_inspection.py`)
- What similarity threshold provides the best balance between accuracy and cache reuse? (`evaluation/threshold_tests.py`)

---

# ⚠️ Limitations

- Small FAQ dataset
- Single embedding model evaluated
- Template-based paraphrases
- Local Ollama inference
- Estimated cost instead of real API billing

---

# 🚀 Future Improvements

- Support larger datasets
- Dynamic threshold selection
- Redis / Vector Database integration
- Production deployment
- Multiple embedding models
- Automatic cache invalidation

---

# 📄 License

This project is licensed under the MIT License.

---

# 👩‍💻 Authors

- Manya Megharaj
- Tejas Hosahalli Devaiah
- Pradeep Singh Yadav
- Yashu Bopanna Pasura Devaiah
- Abhishek Gowda Thammanna Gowda

---

⭐ If you found this project useful, consider giving it a star!
