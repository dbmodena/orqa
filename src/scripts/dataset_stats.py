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
    plots_path = root_path / "plots"

    if not datasets_path.exists():
        print(f"Error: Datasets directory not found at {datasets_path}")
        return

    plots_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(datasets_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {datasets_path}")
        return

    print(f"Processing {len(csv_files)} datasets...")

    all_stats = []
    for csv_file in csv_files:
        stats = get_dataset_stats(csv_file)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("No statistics collected.")
        return

    # Create summary DataFrame
    summary_df = pl.DataFrame(all_stats)

    # Save summary
    summary_csv = plots_path / "datasets_summary.csv"
    summary_df.write_csv(summary_csv)
    print(f"Summary saved to {summary_csv}")

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
            )
        )

    # Attempt plotting
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("\nGenerating plots...")

        # Plot 1: Row/Col counts (Raw vs Clean)
        plt.figure(figsize=(10, 6))
        # This is a bit complex for a simple script, let's just do Row counts raw vs clean
        plot_df = summary_df.select(["filename", "raw_rows", "clean_rows"]).to_pandas()
        plot_df = plot_df.melt(id_vars="filename", var_name="State", value_name="Rows")

        sns.barplot(data=plot_df, x="filename", y="Rows", hue="State")
        plt.xticks(rotation=45, ha="right")
        plt.title("Number of Rows: Raw vs Cleaned")
        plt.tight_layout()
        plt.savefig(plots_path / "rows_comparison.png")
        plt.close()

        # Plot 2: Memory usage
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary_df.to_pandas(), x="filename", y="memory_mb")
        plt.xticks(rotation=45, ha="right")
        plt.title("Memory Usage (MB)")
        plt.tight_layout()
        plt.savefig(plots_path / "memory_usage.png")
        plt.close()

        print(f"Plots saved to {plots_path}")

    except ImportError:
        print("\n[WARNING] matplotlib or seaborn not found. Skipping plot generation.")
        print(
            "To enable plots, install them in your environment: pip install matplotlib seaborn"
        )


if __name__ == "__main__":
    main()
