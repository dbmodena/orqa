import argparse
import os
from pathlib import Path
import polars as pl
import sys

# Add src to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orqa.utils import remove_null_rows, remove_null_columns


def get_dataset_stats(file_path: Path):
    """
    Get statistics for a single dataset.
    """
    try:
        # Read-only mode: read_csv is naturally read-only unless we write back
        df_raw = pl.read_csv(file_path, ignore_errors=True)

        raw_rows, raw_cols = df_raw.shape

        # Cleaning
        df_clean = remove_null_columns(df_raw)
        df_clean = remove_null_rows(df_clean)

        clean_rows, clean_cols = df_clean.shape

        # Type distribution
        type_counts = {}
        for dtype in df_clean.dtypes:
            dtype_str = str(dtype)
            type_counts[dtype_str] = type_counts.get(dtype_str, 0) + 1

        # Additional info
        memory_usage = df_raw.estimated_size("mb")
        total_nulls = df_raw.null_count().sum_horizontal().sum()
        total_cells = raw_rows * raw_cols
        sparsity = (total_nulls / total_cells) if total_cells > 0 else 0

        return {
            "filename": file_path.name,
            "raw_rows": raw_rows,
            "raw_cols": raw_cols,
            "clean_rows": clean_rows,
            "clean_cols": clean_cols,
            "type_counts": type_counts,
            "memory_mb": round(memory_usage, 2),
            "sparsity": round(sparsity, 4),
            "total_nulls": int(total_nulls),
        }
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Get statistics about datasets in a given path."
    )
    parser.add_argument(
        "--root", type=str, default="~/data/orqa/ckan/uk", help="Root path of the data"
    )
    args = parser.parse_args()

    root_path = Path(os.path.expanduser(args.root))
    datasets_path = root_path / "datasets" / "crawling" / "csv"
    plots_path = root_path / "statistics"

    if not datasets_path.exists():
        print(f"Error: Datasets directory not found at {datasets_path}")
        return

    plots_path.mkdir(parents=True, exist_ok=True)

    # csv_files = list(datasets_path.glob("*.csv"))
    # if not csv_files:
    #     print(f"No CSV files found in {datasets_path}")
    #     return
    #
    # print(f"Processing {len(csv_files)} datasets...")
    #
    # all_stats = []
    # for csv_file in csv_files:
    #     stats = get_dataset_stats(csv_file)
    #     if stats:
    #         all_stats.append(stats)
    #
    # if not all_stats:
    #     print("No statistics collected.")
    #     return

    # Create summary DataFrame
    # summary_df = pl.DataFrame(all_stats)
    summary_csv_path = plots_path / "datasets_stats.csv"
    if not summary_csv_path.exists():
        print(f"Error: {summary_csv_path} not found. Please run the script once with dataset processing enabled.")
        return
    
    summary_df = pl.read_csv(summary_csv_path)

    # Add derived metrics for cleaning impact
    summary_df = summary_df.with_columns([
        (pl.col("raw_rows") - pl.col("clean_rows")).alias("rows_removed"),
        ((pl.col("raw_rows") - pl.col("clean_rows")) / pl.col("raw_rows") * 100).fill_nan(0).alias("pct_rows_removed"),
        (pl.col("raw_cols") - pl.col("clean_cols")).alias("cols_removed"),
        ((pl.col("raw_cols") - pl.col("clean_cols")) / pl.col("raw_cols") * 100).fill_nan(0).alias("pct_cols_removed"),
    ])

    # Print summary table
    with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True):
        print("\n### Datasets Summary Statistics")
        print(
            summary_df.select(
                [
                    "filename",
                    "raw_rows",
                    "raw_cols",
                    "clean_rows",
                    "clean_cols",
                    "memory_mb",
                    "sparsity",
                ]
            ).head(20) # Show only top 20 for brevity
        )

    # Aggregated Summary
    agg_stats = summary_df.select([
        pl.col("raw_rows").mean().alias("avg_raw_rows"),
        pl.col("clean_rows").mean().alias("avg_clean_rows"),
        pl.col("pct_rows_removed").mean().alias("avg_pct_rows_removed"),
        pl.col("pct_cols_removed").mean().alias("avg_pct_cols_removed"),
        pl.col("memory_mb").mean().alias("avg_memory_mb"),
        pl.col("sparsity").mean().alias("avg_sparsity"),
    ])
    
    print("\n### Aggregated Metrics (Across all datasets)")
    print(agg_stats)

    # Attempt plotting
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("\nGenerating plots...")
        sns.set_theme(style="whitegrid")

        # Plot 1: Distribution of Rows (Raw vs Clean)
        plt.figure(figsize=(10, 6))
        plot_df = summary_df.select(["raw_rows", "clean_rows"]).to_pandas()
        plot_df = plot_df.melt(var_name="State", value_name="Rows")
        sns.boxplot(data=plot_df, x="State", y="Rows")
        plt.yscale("log")
        plt.title("Distribution of Row Counts (Log Scale)")
        plt.tight_layout()
        plt.savefig(plots_path / "rows_distribution.png")
        plt.close()

        # Plot 2: Cleaning Impact (Percentage removed)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(summary_df["pct_rows_removed"].to_pandas(), bins=20, kde=True)
        plt.title("% Rows Removed by Cleaning")
        plt.yscale("log")
        plt.xlabel("Percentage")

        plt.subplot(1, 2, 2)
        sns.histplot(summary_df["pct_cols_removed"].to_pandas(), bins=20, kde=True)
        plt.title("% Columns Removed by Cleaning")
        plt.yscale("log")
        plt.xlabel("Percentage")
        
        plt.tight_layout()
        plt.savefig(plots_path / "cleaning_impact_dist.png")
        plt.close()

        # Plot 3: Sparsity and Memory Distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(summary_df["sparsity"].to_pandas(), bins=20, color="green", kde=True)
        plt.title("Sparsity Distribution")
        plt.yscale("log")
        plt.xlabel("Sparsity")
        
        plt.subplot(1, 2, 2)
        sns.histplot(summary_df["memory_mb"].to_pandas(), bins=20, color="orange", kde=True)
        plt.title("Memory Usage Distribution (MB)")
        plt.yscale("log")
        plt.xlabel("MB")
        
        plt.tight_layout()
        plt.savefig(plots_path / "metrics_distribution.png")
        plt.close()

        # Plot 4: Individual comparison (only if few datasets)
        if len(summary_df) <= 30:
            plt.figure(figsize=(12, 6))
            plot_df = summary_df.select(["filename", "raw_rows", "clean_rows"]).to_pandas()
            plot_df = plot_df.melt(id_vars="filename", var_name="State", value_name="Rows")
            sns.barplot(data=plot_df, x="filename", y="Rows", hue="State")
            plt.xticks(rotation=45, ha="right")
            plt.title("Row Comparison: Raw vs Cleaned")
            plt.tight_layout()
            plt.yscale("log")
            plt.savefig(plots_path / "rows_comparison_bar.png")
            plt.close()
        else:
            print(f"Skipping individual bar plots as there are {len(summary_df)} datasets.")

        print(f"Aggregated plots saved to {plots_path}")

    except ImportError:
        print("\n[WARNING] matplotlib or seaborn not found. Skipping plot generation.")
        print(
            "To enable plots, install them in your environment: pip install matplotlib seaborn"
        )


if __name__ == "__main__":
    main()
