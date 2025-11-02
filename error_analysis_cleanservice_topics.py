import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
import os



def read_errors_from_file(file_path):
    """
    Reads a file containing retrieval errors in the given format and
    returns a list of dictionaries with keys:
    - index
    - question
    - true_topic
    - retrieved_topics (list)
    """
    errors = []
    
    # Pattern to match each error entry
    pattern = re.compile(
        r"Index\s+(\d+)\s*-\s*Question:\s*(.*?)\s*True Topic:([^R]+?)\s*Retrieved topics:\s*(\[.*?\])",
        re.DOTALL
    )

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    matches = pattern.findall(text)
    
    for match in matches:
        idx = int(match[0])
        question = match[1].strip()
        true_topic = match[2].strip()
        retrieved_topics = ast.literal_eval(match[3])  # safely parse list string
        
        errors.append({
            "index": idx,
            "question": question,
            "true_topic": true_topic,
            "retrieved_topics": retrieved_topics
        })
    
    return errors

if __name__ == '__main__':

    # =========================
    # 1. Load topics.csv (for topic list)
    # =========================
    topics_df = pd.read_csv("data/clearservice/topics.csv")
    topics = topics_df["Topic"].tolist()

    # =========================
    # 2. Process each log file separately
    # =========================

    FOLDERNAME = 'logs'
    for item in os.listdir(FOLDERNAME):
        item_name = item[0:len(item) - 3]
        item_path = os.path.join(FOLDERNAME, item)

        print('LOG file: ' + item)
        errors = read_errors_from_file(item_path)
        print(errors)

        # =========================
        # 3. Create confusion matrix
        # =========================
        conf_matrix = pd.DataFrame(0, index=topics, columns=topics)

        for err in errors:
            for retrieved in err["retrieved_topics"]:
                conf_matrix.loc[err["true_topic"], retrieved] += 1

        # =========================
        # 4. Plot confusion matrix
        # =========================
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=True)
        plt.title(item + " (Recall@3 Errors)")
        plt.xlabel("Retrieved Topic")
        plt.ylabel("True Topic")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig('figures/' + item_name)
        plt.show()

        # =========================
        # 5. Summarize most frequent wrong topics per true topic
        # =========================
        summary = (
            conf_matrix
            .stack()
            .reset_index()
            .rename(columns={"level_0": "True_Topic", "level_1": "Retrieved_Topic", 0: "Count"})
            .query("Count > 0 & True_Topic != Retrieved_Topic")
            .sort_values(["Count", "True_Topic"], ascending=[False, True])
        )

        print("\nMost frequent wrong retrievals:")
        print(summary)
