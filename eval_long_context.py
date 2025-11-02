import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# --- OpenAI ---
from openai import OpenAI

# --- Google Gemini ---
import google.generativeai as genai


# ====================================================
# 1. Load topics and QA data
# ====================================================
topics_df = pd.read_csv("data/clearservice/topics.csv")
topics = topics_df["Topic"].tolist()

df = pd.read_csv("data/clearservice/cs_qa.csv")
print(f"Clearservice data loaded: {len(df)} items.")


# ====================================================
# 2. Initialize both clients
# ====================================================
load_dotenv()

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Gemini client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# ====================================================
# 3. Prompt builder
# ====================================================
def build_prompt(question, topics):
    """
    Build a retrieval prompt.
    """
    return f"""
You are a retrieval assistant. You are given a list of topics and a question.
Select the **three most relevant topics** that best answer the question.

Return ONLY a numbered list of topic names, in order of relevance.

# Topics:
{chr(10).join([f"- {t}" for t in topics])}

# Question:
{question}

# Response format example:
1. Topic A
2. Topic B
3. Topic C
"""


# ====================================================
# 4. Ask the model for topics
# ====================================================
def ask_llm_for_topics(question, topics, model="openai"):
    """
    Ask either OpenAI or Gemini model to return top 3 relevant topics.
    """
    prompt = build_prompt(question, topics)

    if model == "openai":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()

    elif model == "gemini":
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

    else:
        raise ValueError("Unknown model: choose 'openai' or 'gemini'")

    # Extract topic names
    lines = [line.strip("1234567890. ").strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if line in topics][:3]


# ====================================================
# 5. Evaluation metrics
# ====================================================
def evaluate_retrieval(df, topics, model="openai"):
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_3 = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        true_topic = row["topic"]

        retrieved = ask_llm_for_topics(question, topics, model=model)

        if not retrieved:
            reciprocal_ranks.append(0)
            continue

        if true_topic in retrieved:
            rank = retrieved.index(true_topic) + 1
            reciprocal_ranks.append(1 / rank)
            if rank == 1:
                recall_at_1 += 1
            recall_at_3 += 1
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / len(df)
    recall1 = recall_at_1 / len(df)
    recall3 = recall_at_3 / len(df)
    return mrr, recall1, recall3


# ====================================================
# 6. Run both evaluations
# ====================================================
if __name__ == "__main__":
    # print("\n=== Evaluating OpenAI (gpt-4o-mini) ===")
    # mrr_o, r1_o, r3_o = evaluate_retrieval(df, topics, model="openai")
    # print(f"MRR: {mrr_o:.3f} | Recall@1: {r1_o:.3f} | Recall@3: {r3_o:.3f}")

    print("\n=== Evaluating Gemini (gemini-2.5-flash) ===")
    mrr_g, r1_g, r3_g = evaluate_retrieval(df, topics, model="gemini")
    print(f"MRR: {mrr_g:.3f} | Recall@1: {r1_g:.3f} | Recall@3: {r3_g:.3f}")
