import os
import re
import ast
import pandas as pd

def parse_error_log(filepath):
    """
    Parse a log file into a list of (index, question) tuples where Recall@3 failed.
    """
    pattern = re.compile(
        r"Index\s+(\d+)\s*-\s*Question:\s*(.*?)\s*True Topic:([^R]+?)\s*Retrieved topics:\s*(\[.*?\])",
        re.DOTALL
    )
    errors = []
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    matches = pattern.findall(text)
    for match in matches:
        idx = int(match[0])
        question = match[1].strip()
        true_topic = match[2].strip()
        retrieved_topics = ast.literal_eval(match[3])
        errors.append({
            "index": idx,
            "question": question,
            "true_topic": true_topic,
            "retrieved_topics": retrieved_topics
        })
    return errors

def compute_error_rates(logs_folder, output_csv):
    """
    Reads all logs in logs_folder, computes error rates per question.
    """
    all_files = [f for f in os.listdir(logs_folder) if os.path.isfile(os.path.join(logs_folder, f))]
    
    all_errors = []
    embedder_names = []
    
    for file in all_files:
        filepath = os.path.join(logs_folder, file)
        embedder_name = os.path.splitext(file)[0]
        embedder_names.append(embedder_name)
        errors = parse_error_log(filepath)
        for err in errors:
            all_errors.append((embedder_name, err["index"], err["question"]))
    
    # Convert to DataFrame
    df = pd.DataFrame(all_errors, columns=["embedder", "index", "question"])
    
    # Count errors per question
    error_counts = df.groupby(["index", "question"]).size().reset_index(name="error_count")
    
    # Compute error rate
    total_embedders = len(embedder_names)
    error_counts["error_rate"] = error_counts["error_count"] / total_embedders
    
    # Sort by error rate (desc) then by index
    error_counts = error_counts.sort_values(["error_rate", "index"], ascending=[False, True])
    
    # Save to CSV
    error_counts.to_csv(output_csv, index=False, encoding="utf-8")
    
    print(f"✅ Error rate report saved to {output_csv}")
    return error_counts

# Example usage:
logs_folder = "logs"
os.makedirs("out", exist_ok=True)
output_csv = "out/errors.csv"
report_df = compute_error_rates(logs_folder, output_csv)
print(report_df)
