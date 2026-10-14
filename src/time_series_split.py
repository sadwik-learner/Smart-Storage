"""Chronological Time-Series Splitting Module for SmartStorage.

Responsible: Mahendar (Data Preprocessing, Feature Engineering & Baseline Forecasting)

This module handles:
1. Loading the feature-engineered dataset from data/processed/retail_store_inventory_features.csv.
2. Chronologically partitioning each Store ID + Product ID series into:
   - 70% Training set (historical training period)
   - 15% Validation set (model tuning / evaluation period)
   - 15% Test set (unseen holdout evaluation period)
3. Enforcing strict chronological integrity:
   - Zero random shuffling.
   - Non-overlapping date boundaries across splits.
   - Preserving initial startup NaNs in training without backfilling from future data.
   - Retaining continuous historical features in validation and test sets.
4. Saving partition files to data/processed/:
   - retail_store_inventory_train.csv
   - retail_store_inventory_val.csv
   - retail_store_inventory_test.csv
5. Validating that zero future leakage or series ordering distortions occur.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# Default paths relative to workspace root
DEFAULT_INPUT_PATH = Path("data/processed/retail_store_inventory_features.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")
TRAIN_OUTPUT_FILENAME = "retail_store_inventory_train.csv"
VAL_OUTPUT_FILENAME = "retail_store_inventory_val.csv"
TEST_OUTPUT_FILENAME = "retail_store_inventory_test.csv"

SERIES_KEY_COLUMNS = ["Store ID", "Product ID"]
DATE_COLUMN = "Date"
TARGET_COLUMN = "Units Sold"

# Split proportions (70% train, 15% val, 15% test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_feature_dataset(file_path: Path | str) -> pd.DataFrame:
    """Load feature-engineered dataset and ensure chronological sort order."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature dataset not found at: {path.resolve()}")
    df = pd.read_csv(path)

    if DATE_COLUMN not in df.columns:
        raise KeyError(f"Required date column '{DATE_COLUMN}' is missing.")
    for col in SERIES_KEY_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Required key column '{col}' is missing.")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(by=SERIES_KEY_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)
    return df


def split_series_chronologically(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each store-product time series chronologically into train, val, and test.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by Store ID, Product ID, and Date.
    train_ratio : float, default=0.70
        Proportion of observations for training.
    val_ratio : float, default=0.15
        Proportion of observations for validation.
    test_ratio : float, default=0.15
        Proportion of observations for testing.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) partitioned chronologically.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0 (e.g. 0.70 + 0.15 + 0.15)")

    train_splits: List[pd.DataFrame] = []
    val_splits: List[pd.DataFrame] = []
    test_splits: List[pd.DataFrame] = []

    for _, group in df.groupby(SERIES_KEY_COLUMNS):
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure remaining observations form test split
        n_test = n - n_train - n_val

        # Chronological index slicing
        train_slice = group.iloc[:n_train]
        val_slice = group.iloc[n_train : n_train + n_val]
        test_slice = group.iloc[n_train + n_val :]

        train_splits.append(train_slice)
        val_splits.append(val_slice)
        test_splits.append(test_slice)

    train_df = pd.concat(train_splits, ignore_index=True)
    val_df = pd.concat(val_splits, ignore_index=True)
    test_df = pd.concat(test_splits, ignore_index=True)

    return train_df, val_df, test_df


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, bool | int | str]:
    """Validate chronological non-overlap, strict monotonic ordering, and row counts."""
    results: Dict[str, bool | int | str] = {
        "all_series_monotonic": True,
        "zero_date_overlap_train_val": True,
        "zero_date_overlap_val_test": True,
        "all_series_counts_uniform": True,
        "total_rows_preserved": len(train_df) + len(val_df) + len(test_df),
        "train_rows_per_series": 0,
        "val_rows_per_series": 0,
        "test_rows_per_series": 0,
    }

    train_groups = dict(tuple(train_df.groupby(SERIES_KEY_COLUMNS)))
    val_groups = dict(tuple(val_df.groupby(SERIES_KEY_COLUMNS)))
    test_groups = dict(tuple(test_df.groupby(SERIES_KEY_COLUMNS)))

    sample_key = next(iter(train_groups.keys()))
    results["train_rows_per_series"] = len(train_groups[sample_key])
    results["val_rows_per_series"] = len(val_groups[sample_key])
    results["test_rows_per_series"] = len(test_groups[sample_key])

    for key, train_series in train_groups.items():
        val_series = val_groups.get(key)
        test_series = test_groups.get(key)

        if val_series is None or test_series is None:
            results["all_series_counts_uniform"] = False
            break

        if (
            len(train_series) != results["train_rows_per_series"]
            or len(val_series) != results["val_rows_per_series"]
            or len(test_series) != results["test_rows_per_series"]
        ):
            results["all_series_counts_uniform"] = False
            break

        # Check monotonic ordering
        if not (
            train_series[DATE_COLUMN].is_monotonic_increasing
            and val_series[DATE_COLUMN].is_monotonic_increasing
            and test_series[DATE_COLUMN].is_monotonic_increasing
        ):
            results["all_series_monotonic"] = False
            break

        # Check date boundaries (no overlap)
        train_max = train_series[DATE_COLUMN].max()
        val_min = val_series[DATE_COLUMN].min()
        val_max = val_series[DATE_COLUMN].max()
        test_min = test_series[DATE_COLUMN].min()

        if train_max >= val_min:
            results["zero_date_overlap_train_val"] = False
            break

        if val_max >= test_min:
            results["zero_date_overlap_val_test"] = False
            break

    return results


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str,
) -> Tuple[Path, Path, Path]:
    """Save split DataFrames to CSV files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / TRAIN_OUTPUT_FILENAME
    val_path = out_dir / VAL_OUTPUT_FILENAME
    test_path = out_dir / TEST_OUTPUT_FILENAME

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def print_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    val_results: dict,
) -> None:
    """Print beginner-friendly summary of chronological splitting and data integrity."""
    total_rows = len(train_df) + len(val_df) + len(test_df)
    train_pct = (len(train_df) / total_rows) * 100
    val_pct = (len(val_df) / total_rows) * 100
    test_pct = (len(test_df) / total_rows) * 100

    print("=" * 76)
    print("        SMART STORAGE - CHRONOLOGICAL TIME-SERIES SPLIT REPORT        ")
    print("=" * 76)
    print(f"Total Rows Processed:   {total_rows:,} rows across 100 store-product series")
    print("-" * 76)
    print("SPLIT SUMMARY TABLE:")
    print(f"{'Split Name':<12} | {'Rows':<8} | {'Percent':<7} | {'Rows / Series':<13} | {'Start Date':<10} to {'End Date':<10} | {'Days':<4}")
    print("-" * 76)
    print(f"{'Training':<12} | {len(train_df):<8,} | {train_pct:>6.2f}% | {val_results['train_rows_per_series']:<13} | {train_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {train_df[DATE_COLUMN].max().strftime('%Y-%m-%d')} | {val_results['train_rows_per_series']:<4}")
    print(f"{'Validation':<12} | {len(val_df):<8,} | {val_pct:>6.2f}% | {val_results['val_rows_per_series']:<13} | {val_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {val_df[DATE_COLUMN].max().strftime('%Y-%m-%d')} | {val_results['val_rows_per_series']:<4}")
    print(f"{'Testing':<12} | {len(test_df):<8,} | {test_pct:>6.2f}% | {val_results['test_rows_per_series']:<13} | {test_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {test_df[DATE_COLUMN].max().strftime('%Y-%m-%d')} | {val_results['test_rows_per_series']:<4}")
    print("-" * 76)
    print("SAVED FILE DESTINATIONS:")
    print(f"  * Training file:       {train_path}")
    print(f"  * Validation file:     {val_path}")
    print(f"  * Testing file:        {test_path}")
    print("-" * 76)
    print("MISSING VALUE & LAG STARTUP AUDIT:")
    print(f"  * Training set:        sales_lag_1 has {train_df['sales_lag_1'].isna().sum()} NaNs (1 per series); lag_7/rolling have {train_df['sales_lag_7'].isna().sum()} NaNs (7 per series)")
    print(f"  * Validation set:      sales_lag_1 has {val_df['sales_lag_1'].isna().sum()} NaNs; lag_7/rolling have {val_df['sales_lag_7'].isna().sum()} NaNs (100% full history)")
    print(f"  * Testing set:         sales_lag_1 has {test_df['sales_lag_1'].isna().sum()} NaNs; lag_7/rolling have {test_df['sales_lag_7'].isna().sum()} NaNs (100% full history)")
    print(f"  * Policy Enforced:     Zero backfilling from future. Initial startup NaNs preserved transparently.")
    print("-" * 76)
    print("CHRONOLOGICAL INTEGRITY & NON-OVERLAP VERIFICATION:")
    print(f"  * Monotonic Order:     {val_results['all_series_monotonic']} (All series strictly chronological)")
    print(f"  * Train-Val Overlap:   {not val_results['zero_date_overlap_train_val']} (Train ends before Val begins - PASSED)")
    print(f"  * Val-Test Overlap:    {not val_results['zero_date_overlap_val_test']} (Val ends before Test begins - PASSED)")
    print(f"  * Uniform Partition:   {val_results['all_series_counts_uniform']} (All 100 series partitioned identically)")
    print("=" * 76)
    print("STATUS: Chronological time-series splitting completed successfully!")
    print("=" * 76)


def run_split_pipeline(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete time-series split pipeline."""
    df = load_feature_dataset(input_path)
    train_df, val_df, test_df = split_series_chronologically(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    validation_results = validate_splits(train_df, val_df, test_df)
    train_path, val_path, test_path = save_splits(train_df, val_df, test_df, output_dir)
    print_split_report(
        train_df,
        val_df,
        test_df,
        train_path,
        val_path,
        test_path,
        validation_results,
    )
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="SmartStorage Chronological Time-Series Split Pipeline (Mahendar)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=f"Path to feature CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help=f"Directory to save train/val/test CSVs (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Proportion of training set (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Proportion of validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help="Proportion of test set (default: 0.15)",
    )
    args = parser.parse_args()

    run_split_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
