import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            # Flatten the metrics or handle missing ones
            metrics = record.get("metrics", {})
            row = {
                "task": record.get("task", "unknown"),
                "overlap_ratio": metrics.get("overlap_ratio"),
                "sm_micro_avg": metrics.get("sm_micro_avg"),
                "sm_macro_avg": metrics.get("sm_macro_avg"),
                "sm_n_matches": metrics.get("sm_n_matches"),
            }
            data.append(row)
    return pd.DataFrame(data)


def plot_stats(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_theme(style="whitegrid", palette="viridis")

    # Metrics to plot
    metrics = ["overlap_ratio", "sm_micro_avg", "sm_macro_avg", "sm_n_matches"]

    # 1. Bar plots for averages
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="task", y=metric, errorbar="sd")
        plt.title(f"Average {metric.replace('_', ' ').title()} by Task Type")
        plt.xlabel("Task Type")
        plt.ylabel(metric.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(output_dir / f"avg_{metric}.png")
        plt.close()

    # 2. Distribution plots
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="task", y=metric)
        plt.title(f"Distribution of {metric.replace('_', ' ').title()} by Task Type")
        plt.xlabel("Task Type")
        plt.ylabel(metric.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(output_dir / f"dist_{metric}.png")
        plt.close()

    # 3. Record counts
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="task")
    plt.title("Number of Records per Task Type")
    plt.xlabel("Task Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "record_counts.png")
    plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    input_file = (
        "/home/nanni/data/orqa/socrata/nyc/candidates_discovery/tasks_results.json"
    )
    output_folder = "/home/nanni/projects/orqa/plots/task_stats"

    print(f"Loading data from {input_file}...")
    df = load_data(input_file)

    # Filter out rows with no metrics if necessary for some plots
    # (Though boxplots/barplots handle NaNs gracefully by default)

    print("Generating statistics...")
    stats = df.groupby("task").agg(
        {
            "overlap_ratio": ["mean", "std", "count"],
            "sm_micro_avg": ["mean", "std"],
            "sm_macro_avg": ["mean", "std"],
            "sm_n_matches": ["mean", "sum"],
        }
    )
    print(stats)

    print("Creating plots...")
    plot_stats(df, output_folder)
