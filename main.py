import time
import pandas as pd

from embeddings.encoder import generate_embedding
from cache.exact_cache import (
    get_exact_match,
    store_exact_match
)
from cache.semantic_cache import (
    store_semantic,
    search_semantic,
    questions,
    answers
)
from llm.ollama_client import ask_llm


# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("data/university_faq_dataset.csv")

print("Dataset Loaded:", df.shape)


# ============================================
# BUILD SEMANTIC CACHE FROM DATASET
# ============================================

print("\nGenerating embeddings and building FAISS index...\n")

for _, row in df.iterrows():

    question = row["question"]
    answer = row["answer"]

    # Store in exact cache
    store_exact_match(question, answer)

    # Generate embedding
    embedding = generate_embedding(question)

    # Store in semantic cache
    store_semantic(
        question,
        embedding,
        answer
    )

print("Semantic cache ready.")
print("Total Questions Indexed:", len(questions))


# ============================================
# RETRIEVE CONTEXT
# ============================================

def retrieve_context(query):

    query_lower = query.lower()

    for _, row in df.iterrows():

        if query_lower in row["question"].lower():

            return row["answer"]

    return "No relevant context found."


# ============================================
# MAIN QUERY PIPELINE
# ============================================

def process_query(query, threshold=0.5):

    start_time = time.time()

    print("\n===================================")
    print("USER QUERY:", query)
    print("===================================\n")

    # ----------------------------------------
    # 1. EXACT CACHE
    # ----------------------------------------

    exact_answer = get_exact_match(query)

    if exact_answer:

        latency = time.time() - start_time

        print("EXACT CACHE HIT")
        print("Latency:", round(latency, 4), "seconds")

        return {
            "source": "exact_cache",
            "answer": exact_answer,
            "latency": latency
        }

    # ----------------------------------------
    # 2. SEMANTIC CACHE
    # ----------------------------------------


    query_embedding = generate_embedding(query)

    scores, indices = search_semantic(query_embedding)

    # scores from IndexFlatIP are inner-products of normalized vectors = cosine
    nearest_score = float(scores[0][0]) if scores.size else 0.0

    matched_index = int(indices[0][0]) if indices.size else None

    matched_question = questions[matched_index] if matched_index is not None else None

    matched_answer = answers[matched_index] if matched_index is not None else None

    print("Nearest Match:", matched_question)
    print("Similarity Score:", round(nearest_score, 4))

    if nearest_score > threshold:

        latency = time.time() - start_time

        print("\nSEMANTIC CACHE HIT")
        print("Latency:", round(latency, 4), "seconds")

        return {
            "source": "semantic_cache",
            "answer": matched_answer,
            "matched_question": matched_question,
            "similarity": nearest_score,
            "latency": latency
        }

    # ----------------------------------------
    # 3. LLM FALLBACK
    # ----------------------------------------

    print("\nCACHE MISS -> CALLING LLM...\n")

    context = retrieve_context(query)

    llm_answer = ask_llm(
        question=query,
        context=context
    )

    # ----------------------------------------
    # STORE NEW RESULT
    # ----------------------------------------

    store_exact_match(query, llm_answer)

    store_semantic(
        query,
        query_embedding,
        llm_answer
    )

    latency = time.time() - start_time

    print("LLM RESPONSE GENERATED")
    print("Latency:", round(latency, 4), "seconds")

    return {
        "source": "llm",
        "answer": llm_answer,
        "latency": latency
    }


# ============================================
# INTERACTIVE LOOP
# ============================================

if __name__ == "__main__":
    print("\n===================================")
    print("Semantic Caching System Ready")
    print("Type 'exit' to quit")
    print("===================================\n")


    while True:

        user_query = input("\nAsk Question: ")

        if user_query.lower() == "exit":
            print("\nExiting system...")
            break

        result = process_query(user_query)

        print("\n-----------------------------------")
        print("SOURCE:", result["source"])
        print("-----------------------------------")

        if "similarity" in result:
            print("SIMILARITY:", round(result["similarity"], 4))

        print("\nANSWER:\n")
        print(result["answer"])
        print("\n-----------------------------------")