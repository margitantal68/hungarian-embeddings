import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def plot_model_comparison():
    # Data
    data = {
    "Model": [
        "BGE-M3",
        "E5-BASE",
        "GEMINI",
        "HUBERT",
        "NOMIC",
        "OPENAI-3SMALL",
        "OPENAI-ADA",
        "XLMROBERTA",
        "BM25"
    ],
    "MRR": [0.90, 0.79, 0.87, 0.78, 0.71, 0.80, 0.80, 0.90, 0.77],
    "Recall@1": [0.86, 0.70, 0.78, 0.74, 0.64, 0.70, 0.72, 0.86, 0.68],
    "Recall@3": [0.96, 0.92, 0.98, 0.84, 0.80, 0.94, 0.90, 0.96, 0.80]
    }


    df = pd.DataFrame(data)
    bar_width = 0.2
    x = range(len(df))

    # Bar positions
    positions_mrr = [i - bar_width for i in x]
    positions_r1 = x
    positions_r3 = [i + bar_width for i in x]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(positions_mrr, df["MRR"], width=bar_width, label="MRR", color="#1f77b4")
    ax.bar(positions_r1, df["Recall@1"], width=bar_width, label="Recall@1", color="#ff7f0e")
    ax.bar(positions_r3, df["Recall@3"], width=bar_width, label="Recall@3", color="#2ca02c")

    # Labels and titles
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: MRR, Recall@1, Recall@3")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()



def plot_mrr_clearservice():
    data = {
        "Model": [
            "BGE-M3",
            "E5-BASE",
            "GEMINI",
            "HUBERT",
            "NOMIC",
            "OPENAI-3SMALL",
            "OPENAI-ADA",
            "XLMROBERTA",
            "BM25"
        ],
        # "Recall@3": [0.96, 0.92, 0.98, 0.84, 0.80, 0.94, 0.90, 0.96, 0.80]
        "MRR": [0.90, 0.79, 0.87, 0.78, 0.71, 0.80, 0.80, 0.90, 0.77]
    }

    models = data["Model"][::-1]       # Reverse the list
    recall_values = data["MRR"][::-1]  # Reverse corresponding values

    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, recall_values, color='skyblue')
    plt.xlabel("MRR")
    plt.title("Clearservice dataset")
    plt.xlim(0, 1.05)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', va='center')

    plt.tight_layout()
    plt.show()


def plot_mrr_hurte(data=None):
    # Use provided data or the default HuRTE values
    if data is None:
    #    data = {
    # "Model": [
    #     "BGE-M3",
    #     "E5-BASE",
    #     "GEMINI",
    #     "HUBERT",
    #     "NOMIC",
    #     "OPENAI-3SMALL",
    #     "OPENAI-ADA",
    #     "XLMROBERTA",
    #     "BM25"
    # ],
    # "training":    [0.97, 0.92, 0.98, 0.74, 0.80, 0.92, 0.92, 0.91, 0.79],
    # "development": [1.00, 0.97, 1.00, 0.88, 0.95, 0.97, 0.98, 0.98, 0.84]
    # }

        data = {
        "Model": [
            "BGE-M3",
            "E5-BASE",
            "GEMINI",
            "HUBERT",
            "NOMIC",
            "OPENAI-3SMALL",
            "OPENAI-ADA",
            "XLMROBERTA",
            "BM25"
        ],
        "training": [0.89, 0.84, 0.91, 0.63, 0.72, 0.85, 0.84, 0.82, 0.72],
        "development": [0.98, 0.93, 0.99, 0.82, 0.90, 0.94, 0.94, 0.94, 0.82]
    }


    models = list(data["Model"])
    training_values = list(data["training"])
    dev_values = list(data["development"])

    # Keep the original order — no sorting
    y = np.arange(len(models))
    bar_h = 0.38

    plt.figure(figsize=(10, 6))
    bars_train = plt.barh(y - bar_h/2, training_values, height=bar_h, label="train", color="skyblue")
    bars_dev   = plt.barh(y + bar_h/2, dev_values,    height=bar_h, label="dev", color="lightgreen")

    plt.yticks(y, models)
    plt.xlabel("MRR")
    plt.title("HuRTE dataset: Training vs Development")
    plt.xlim(0, 1.05)
    plt.legend()

    # Annotate bar values
    for bars in (bars_train, bars_dev):
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va="center")

    # Put the first listed model at the top
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()




def plot_hurte_all_recall1():
    data = {
        "Model": [
            "BGE-M3", "E5-BASE", "GEMINI", "HUBERT", "NOMIC",
            "OPENAI-3SMALL", "OPENAI-ADA", "XLMROBERTA", "BM25"
        ],
        "HuRTE-positive": [0.96, 0.90, 0.97, 0.77, 0.85, 0.92, 0.91, 0.91, 0.78],  # Recall@1
        "HuRTE-all":      [0.94, 0.87, 0.96, 0.70, 0.81, 0.90, 0.89, 0.89, 0.75]   # Recall@1
    }

    models = list(data["Model"])
    positive_values = list(data["HuRTE-positive"])
    all_values = list(data["HuRTE-all"])

    y = np.arange(len(models))
    bar_h = 0.38

    plt.figure(figsize=(10, 6))
    bars_positive = plt.barh(y - bar_h/2, positive_values, height=bar_h, label="HuRTE-Positive", color="skyblue")
    bars_all      = plt.barh(y + bar_h/2, all_values,      height=bar_h, label="HuRTE-All", color="lightgreen")

    plt.yticks(y, models)
    plt.xlabel("Recall@1")
    plt.title("HuRTE dataset: HuRTE-Positive vs HuRTE-All (Recall@1)")
    plt.xlim(0, 1.05)
    plt.legend()

    # Annotate bar values
    for bars in (bars_positive, bars_all):
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va="center")

    # First model at the top
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_hurte_all_recall3():
    data = {
        "Model": [
            "BGE-M3", "E5-BASE", "GEMINI", "HUBERT", "NOMIC",
            "OPENAI-3SMALL", "OPENAI-ADA", "XLMROBERTA", "BM25"
        ],
        "HuRTE-positive": [1.00, 0.97, 1.00, 0.88, 0.95, 0.97, 0.98, 0.98, 0.84],  # Recall@3
        "HuRTE-all":      [0.99, 0.96, 1.00, 0.84, 0.93, 0.96, 0.98, 0.97, 0.84]   # Recall@3
    }

    models = list(data["Model"])
    positive_values = list(data["HuRTE-positive"])
    all_values = list(data["HuRTE-all"])

    y = np.arange(len(models))
    bar_h = 0.38

    plt.figure(figsize=(10, 6))
    bars_positive = plt.barh(y - bar_h/2, positive_values, height=bar_h, label="HuRTE-Positive", color="skyblue")
    bars_all      = plt.barh(y + bar_h/2, all_values,      height=bar_h, label="HuRTE-All", color="lightgreen")

    plt.yticks(y, models)
    plt.xlabel("Recall@3")
    plt.title("HuRTE dataset: HuRTE-Positive vs HuRTE-All")
    plt.xlim(0, 1.05)
    plt.legend()

    # Annotate bar values
    for bars in (bars_positive, bars_all):
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va="center")

    # First model at the top
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()



def plot_hurte_all_mrr():
    data = {
        "Model": [
            "BGE-M3", "E5-BASE", "GEMINI", "HUBERT", "NOMIC",
            "OPENAI-3SMALL", "OPENAI-ADA", "XLMROBERTA", "BM25"
        ],
        "HuRTE-positive": [0.98, 0.93, 0.99, 0.82, 0.90, 0.94, 0.94, 0.94, 0.82],  # MRR
        "HuRTE-all":      [0.97, 0.91, 0.98, 0.76, 0.87, 0.93, 0.93, 0.93, 0.80]   # MRR
    }

    models = list(data["Model"])
    positive_values = list(data["HuRTE-positive"])
    all_values = list(data["HuRTE-all"])

    y = np.arange(len(models))
    bar_h = 0.38

    plt.figure(figsize=(10, 6))
    bars_positive = plt.barh(y - bar_h/2, positive_values, height=bar_h, label="HuRTE-Positive", color="skyblue")
    bars_all      = plt.barh(y + bar_h/2, all_values,      height=bar_h, label="HuRTE-All", color="lightgreen")

    plt.yticks(y, models)
    plt.xlabel("MRR")
    plt.title("HuRTE dataset: HuRTE-Positive vs HuRTE-All (MRR)")
    plt.xlim(0, 1.05)
    plt.legend()

    # Annotate bar values
    for bars in (bars_positive, bars_all):
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va="center")

    # First model at the top
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()



def plot_model_performance():
    # Data
    models = ["BGE-M3", "E5-BASE", "GEMINI", "HUBERT", 
              "NOMIC", "OPENAI-3SMALL", "OPENAI-ADA", "XLMROBERTA"]
    index_build_time = [3.50, 1.35, 2.50, 1.30, 4.25, 1.80, 1.55, 1.05]
    avg_query_latency = [0.0374, 0.0254, 0.4177, 0.0215, 0.0264, 0.2903, 0.2626, 0.0248]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))

    # Bar plot for index build time
    bars = ax1.bar(x, index_build_time, color='skyblue', alpha=0.7, label='Index build time (s)')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Index build time (s)', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha='right')

    # Line plot for avg query latency
    ax2 = ax1.twinx()
    ax2.plot(x, avg_query_latency, color='darkorange', marker='o', linewidth=2, label='Avg. Query latency (s)')
    ax2.set_ylabel('Avg. Query latency (s)', color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Title and legend
    plt.title('Model Performance: Index Build Time vs. Avg. Query Latency', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_rag_efficiency():
    # Data
    models = ["BGE-M3", "E5-BASE", "GEMINI", "HUBERT", 
              "NOMIC", "OPENAI-3SMALL", "OPENAI-ADA", "XLMROBERTA"]
    index_build_time = [1.25, 0.55, 1.15, 0.41, 1.05, 0.96, 0.94, 0.35]
    query_latency = [1.96, 1.35, 23.09, 1.13, 1.30, 16.13, 12.59, 1.33]
    mrr = [0.90, 0.79, 0.87, 0.78, 0.71, 0.80, 0.80, 0.90]

    # Scale bubble sizes (index build time)
    sizes = [t * 300 for t in index_build_time]

    # Create figure
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(query_latency, mrr, s=sizes, alpha=0.7, c=mrr, cmap='viridis', edgecolor='k')

    # Add annotations for model names
    for i, model in enumerate(models):
        plt.text(query_latency[i] * 1.02, mrr[i], model, fontsize=9, va='center')

    # Labels and title
    plt.title("Model Efficiency (Clearservice Dataset)", fontsize=14, fontweight='bold')
    plt.xlabel("Average Query Latency (s) ↓", fontsize=12)
    plt.ylabel("Quality (MRR) ↑", fontsize=12)
    plt.grid(alpha=0.4)

    # Legend for bubble sizes
    for size, label in zip([0.5, 1.0, 1.5], ['0.5 s', '1.0 s', '1.5 s']):
        plt.scatter([], [], s=size*300, c='gray', alpha=0.4, label=f'Index build time: {label}')
    plt.legend(frameon=True, loc='lower right', title='Bubble Size')

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("MRR (Mean Reciprocal Rank)", fontsize=11)

    plt.tight_layout()
    plt.show()


# plot_recall_at_3_clearservice()
# plot_recall_at_3_hurte()
# plot_model_comparison()
# plot_hurte_all_recall3()
# plot_hurte_all_recall1()
# plot_hurte_all_mrr()
# plot_model_performance()
# plot_rag_efficiency()
# plot_mrr_clearservice()
# plot_mrr_hurte()
plot_hurte_all_mrr()