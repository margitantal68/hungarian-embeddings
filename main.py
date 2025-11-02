import os
import re
import json
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


from models import (
    OllamaEmbedder,
    OpenAIEmbedder,
    GeminiEmbedder,
    SentenceTransformerEmbedder,
)


def build_chroma_index(texts, model, collection_name="default_collection", persist_dir=None):
    """
    Build a ChromaDB collection, measure build time, memory usage, and raw embedding size.
    """

    # Create client (persistent or in-memory)
    if persist_dir:
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.Client()

    # Remove collection if it exists
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)


    # Time start
    start = time.time()

    # Create collection
    collection = client.create_collection(name=collection_name)

    # Encode embeddings
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    ids = [f"doc_{i}" for i in range(len(texts))]

    # Add to collection
    collection.add(documents=texts, embeddings=embeddings.tolist(), ids=ids)

    # Compute times
    build_time = time.time() - start

    # Compute theoretical raw embedding size
    n, d = embeddings.shape
    bytes_per_float = embeddings.dtype.itemsize
    raw_size_mb = (n * d * bytes_per_float) / (1024 ** 2)

    print(f"✅ ChromaDB collection '{collection_name}' created with {len(texts)} entries.")
    print(f"⏱️  Index build time: {build_time:.2f} seconds.")
    print(f"💾 Raw embedding size: {raw_size_mb:.2f} MB.")

    # If persistent, also check on-disk index size
    if persist_dir:
        def get_dir_size(path):
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    total += os.path.getsize(os.path.join(dirpath, f))
            return total / (1024 ** 2)
        disk_size = get_dir_size(persist_dir)
        print(f"📀 On-disk ChromaDB store size: {disk_size:.2f} MB.")

    return collection, embeddings


# -------------------------------
# Export ChromaDB
# -------------------------------
import json

def export_chroma_collection(collection, embeddings, filename="collection_export.json"):
    items = []
    for i, doc in enumerate(collection.get()["documents"]):
        items.append({
            "id": collection.get()["ids"][i],
            "text": doc,
            "embedding": embeddings[i]
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Exported collection to {filename}")


# -------------------------------
# RTE Evaluation - using positive premises from HURTE dataset
# -------------------------------


def evaluate_models_rte_positive_data(filename, model):
    # Load RTE data
    with open(filename, 'r', encoding='utf-8') as f:
        rte_data = json.load(f)
    print(f"RTE data loaded: {len(rte_data)} items.")

    # Keep only positive (entailment) items
    rte_data = [item for item in rte_data if str(item.get("label", "")) == "1"]
    print(f"Positive RTE data loaded: {len(rte_data)} items.")

    # Build retrieval index using premises
    data_to_embed = [item['premise'] for item in rte_data]
    collection, _ = build_chroma_index(data_to_embed, model, collection_name="rte_collection")

    k = 3
    recall_at_1 = recall_at_3 = 0
    reciprocal_ranks = []
    total_latency = 0.0

    # Evaluate each hypothesis one by one
    for i, item in enumerate(rte_data):
        hypothesis = item['hypothesis']

        # Measure latency for this single query
        query_start = time.time()

        query_vec = model.encode([hypothesis], convert_to_numpy=True, normalize_embeddings=True)[0]
        results = collection.query(query_embeddings=[query_vec.tolist()], n_results=k)

        query_latency = time.time() - query_start
        total_latency += query_latency

        # Extract retrieved indices
        retrieved_ids = [int(id_.split("_")[1]) for id_ in results["ids"][0]]

        # Compute metrics
        if i == retrieved_ids[0]:
            recall_at_1 += 1
        if i in retrieved_ids:
            recall_at_3 += 1
            rank = retrieved_ids.index(i) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    # Compute final metrics
    num_queries = len(rte_data)
    mrr = sum(reciprocal_ranks) / num_queries
    recall_at_1_score = recall_at_1 / num_queries
    recall_at_3_score = recall_at_3 / num_queries
    avg_latency = total_latency / num_queries

    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Recall@1: {recall_at_1_score:.4f}")
    print(f"Recall@3: {recall_at_3_score:.4f}")
    print(f"Average query latency: {avg_latency:.4f} seconds/query")
    print(f"Total evaluation time: {total_latency:.2f} seconds")

# -------------------------------
# RTE Evaluation - using ALL premises from HURTE dataset
# -------------------------------

def evaluate_models_rte_all_data(filename, model):
    with open(filename, 'r', encoding='utf-8') as f:
        rte_data = json.load(f)
    print(f"RTE data loaded: {len(rte_data)} items.")

    # Separate data for retrieval and evaluation
    all_premises = [item['premise'] for item in rte_data]
    positive_items = [item for item in rte_data if str(item.get("label", "")) == "1"]
    print(f"Positive RTE data for evaluation: {len(positive_items)} items.")

    # Build retrieval index using all premises
    collection, _ = build_chroma_index(all_premises, model, collection_name="rte_collection")

    k = 3
    recall_at_1 = recall_at_3 = 0
    reciprocal_ranks = []
    total_latency = 0.0

    # Process each query individually
    for i, item in enumerate(positive_items):
        hypothesis = item['hypothesis']

        # Measure latency: embedding + retrieval
        query_start = time.time()

        query_vec = model.encode([hypothesis], convert_to_numpy=True, normalize_embeddings=True)[0]
        results = collection.query(query_embeddings=[query_vec.tolist()], n_results=k)

        query_latency = time.time() - query_start
        total_latency += query_latency

        # Retrieved IDs correspond to all premises
        retrieved_ids = [int(id_.split("_")[1]) for id_ in results["ids"][0]]

        # Find the index of the correct premise in the full dataset
        correct_idx = rte_data.index(item)

        # Compute metrics
        if correct_idx == retrieved_ids[0]:
            recall_at_1 += 1
        if correct_idx in retrieved_ids:
            recall_at_3 += 1
            rank = retrieved_ids.index(correct_idx) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    num_queries = len(positive_items)
    mrr = sum(reciprocal_ranks) / num_queries
    recall_at_1_score = recall_at_1 / num_queries
    recall_at_3_score = recall_at_3 / num_queries
    avg_latency = total_latency / num_queries

    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Recall@1: {recall_at_1_score:.4f}")
    print(f"Recall@3: {recall_at_3_score:.4f}")
    print(f"Average query latency: {avg_latency:.4f} seconds/query")
    print(f"Total evaluation time: {total_latency:.2f} seconds")

# -------------------------------
# Clearservice Evaluation
# -------------------------------

def parse_topics(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    topics = []
    chunks = re.split(r"^##\s*", content, flags=re.MULTILINE)
    for chunk in chunks[1:]:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        description = "\n".join(lines[1:]).strip()
        text = f"{title}: {description}"
        topics.append(text)
    return topics


def evaluate_models_clearservice_data(model_name, model):
    log_file = f'logs/{model_name}.log'
    os.makedirs("logs", exist_ok=True)

    topic_chunks = parse_topics("data/clearservice/topics.txt")
    
    print("Creating ChromaDB collection...")
    collection, _ = build_chroma_index(topic_chunks, model, collection_name="clearservice_collection")
    print("ChromaDB index created.")


    df = pd.read_csv("data/clearservice/cs_qa.csv")
    print(f"Clearservice data loaded: {len(df)} items.")

    reciprocal_ranks = []
    recall_at_1 = recall_at_3 = 0
    num_questions = len(df)

    start = time.time() 
    with open(log_file, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            question = row["question"]
            topic = row["topic"]
            query_vec = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

            results = collection.query(query_embeddings=[query_vec], n_results=3)
            top_docs = results["documents"][0]
            result_topics = [doc.split(":", 1)[0].strip() for doc in top_docs]

            if topic == result_topics[0]:
                recall_at_1 += 1
            if topic in result_topics:
                recall_at_3 += 1
                rank = result_topics.index(topic) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0.0)
                f.write(
                    f"Index {idx + 1} - Question: {question}\n"
                    f"True Topic: {topic}\n"
                    f"Retrieved topics: {result_topics}\n\n"
                )

    if num_questions > 0:
        recall_at_1_score = recall_at_1 / num_questions
        recall_at_3_score = recall_at_3 / num_questions
        mrr = sum(reciprocal_ranks) / num_questions
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"Recall@1: {recall_at_1_score:.4f}")
        print(f"Recall@3: {recall_at_3_score:.4f}")
    else:
        print("No questions to evaluate.")
    eval_time = time.time() - start
    print(f"Query latency: {eval_time:.2f} seconds.")

# -------------------------------
# Model definitions
# -------------------------------

models = [
    SentenceTransformerEmbedder("BAAI/bge-m3"),
    SentenceTransformerEmbedder("intfloat/multilingual-e5-base"),
    GeminiEmbedder(),
    SentenceTransformerEmbedder("danieleff/hubert-base-cc-sentence-transformer"),
    OllamaEmbedder("nomic-embed-text:latest"),
    OpenAIEmbedder("text-embedding-3-small"),
    OpenAIEmbedder("text-embedding-ada-002"),
    SentenceTransformerEmbedder("sentence-transformers/paraphrase-xlm-r-multilingual-v1"),
]

model_names = [
    "BGE-M3",
    "E5-BASE",
    "GEMINI",
    "HUBERT",
    "NOMIC",
    "OPENAI-3SMALL",
    "OPENAI-ADA",
    "XLMROBERTA"
]

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    index = 3

    model_name = model_names[index]
    model = models[index]
    print(f"Evaluating model: {model_name}")

    evaluate_models_clearservice_data(model_name, model)

    # evaluate_models_rte_positive_data("data/hurte/rte_train.json", model)
    # evaluate_models_rte_positive_data("data/hurte/rte_dev.json", model)

    # evaluate_models_rte_all_data("data/hurte/rte_train.json", model)
    # evaluate_models_rte_all_data("data/hurte/rte_dev.json", model)
# 
    print("Evaluation completed.")

