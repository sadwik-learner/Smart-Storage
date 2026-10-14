"""Data Preprocessing Module for SmartStorage.

Responsible: Mahendar (Data Preprocessing, Feature Engineering & Baseline Forecasting)

This module handles:
1. Loading the canonical retail store inventory dataset.
2. Converting and validating date fields.
3. Sorting records chronologically by Store ID, Product ID, and Date.
4. Performing data quality checks (missing values, full-row duplicates, key-combination duplicates).
5. Excluding the target-leaking 'Demand Forecast' column.
6. Preserving the target variable 'Units Sold'.
7. Saving the cleaned panel dataset to data/processed/retail_store_inventory_clean.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd


# Default paths relative to repository root
DEFAULT_INPUT_PATH = Path("dataset/retail_store_inventory.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/retail_store_inventory_clean.csv")

# Leaked columns identified during Exploratory Data Analysis (EDA)
LEAKED_COLUMNS = ["Demand Forecast"]

# Key series identifier columns
SERIES_KEY_COLUMNS = ["Store ID", "Product ID", "Date"]
TARGET_COLUMN = "Units Sold"


def load_raw_data(file_path: Path | str) -> pd.DataFrame:
    """Load the raw canonical CSV dataset into a pandas DataFrame."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical dataset not found at: {path.resolve()}")
    df = pd.read_csv(path)
    return df


def validate_and_clean_data(
    df: pd.DataFrame,
    drop_leakage: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Validate, clean, and sort the retail sales data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset loaded from the canonical CSV.
    drop_leakage : bool, default=True
        Whether to exclude known target-leaking columns ('Demand Forecast').

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        A tuple of (cleaned DataFrame, summary report dictionary).
    """
    input_shape = df.shape
    report = {
        "input_shape": input_shape,
        "missing_values": {},
        "duplicate_rows": 0,
        "duplicate_series_keys": 0,
        "invalid_dates": 0,
        "date_min": None,
        "date_max": None,
        "dropped_columns": [],
        "target_column": TARGET_COLUMN,
        "target_valid": False,
    }

    # Work on a copy to ensure original DataFrame is preserved unchanged
    clean_df = df.copy()

    # 1. Missing Values Check
    missing_series = clean_df.isna().sum()
    report["missing_values"] = missing_series[missing_series > 0].to_dict()
    report["total_missing"] = int(missing_series.sum())

    # 2. Duplicate Rows Check
    report["duplicate_rows"] = int(clean_df.duplicated().sum())

    # 3. Date Conversion & Validation
    if "Date" not in clean_df.columns:
        raise KeyError("Required column 'Date' is missing from dataset.")

    original_dates = clean_df["Date"]
    clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    invalid_dates_count = int(clean_df["Date"].isna().sum())
    report["invalid_dates"] = invalid_dates_count

    if invalid_dates_count > 0:
        raise ValueError(
            f"Found {invalid_dates_count} invalid date values in 'Date' column."
        )

    report["date_min"] = clean_df["Date"].min().strftime("%Y-%m-%d")
    report["date_max"] = clean_df["Date"].max().strftime("%Y-%m-%d")
    report["unique_dates"] = int(clean_df["Date"].nunique())

    # 4. Series Key Duplicate Check (Store ID + Product ID + Date)
    missing_keys = [col for col in SERIES_KEY_COLUMNS if col not in clean_df.columns]
    if missing_keys:
        raise KeyError(f"Missing required key columns: {missing_keys}")

    key_duplicates = int(clean_df.duplicated(subset=SERIES_KEY_COLUMNS).sum())
    report["duplicate_series_keys"] = key_duplicates

    # 5. Sort by Store ID, Product ID, and Date
    clean_df = clean_df.sort_values(by=SERIES_KEY_COLUMNS).reset_index(drop=True)

    # 6. Target Variable Validation
    if TARGET_COLUMN not in clean_df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' is missing from dataset.")

    target_series = clean_df[TARGET_COLUMN]
    if (target_series < 0).any():
        raise ValueError(f"Found negative sales values in target column '{TARGET_COLUMN}'.")

    report["target_valid"] = True
    report["target_min"] = int(target_series.min())
    report["target_max"] = int(target_series.max())
    report["target_mean"] = round(float(target_series.mean()), 2)

    # 7. Exclude Leaked Features (Demand Forecast)
    if drop_leakage:
        cols_to_drop = [c for c in LEAKED_COLUMNS if c in clean_df.columns]
        if cols_to_drop:
            clean_df = clean_df.drop(columns=cols_to_drop)
            report["dropped_columns"] = cols_to_drop

    report["output_shape"] = clean_df.shape
    return clean_df, report


def save_processed_data(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Save the cleaned DataFrame to CSV, ensuring parent directories exist."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def print_summary_report(report: dict, input_path: Path, output_path: Path) -> None:
    """Print a beginner-friendly, structured summary of the preprocessing results."""
    print("=" * 70)
    print("           SMART STORAGE - DATA PREPROCESSING REPORT           ")
    print("=" * 70)
    print(f"Input file:           {input_path}")
    print(f"Output file:          {output_path}")
    print(f"Input dimensions:     {report['input_shape'][0]:,} rows x {report['input_shape'][1]} columns")
    print(f"Output dimensions:    {report['output_shape'][0]:,} rows x {report['output_shape'][1]} columns")
    print("-" * 70)
    print("CHRONOLOGICAL ORDER & DATES:")
    print(f"  * Sorted by:         Store ID -> Product ID -> Date")
    print(f"  * Date Range:        {report['date_min']} to {report['date_max']}")
    print(f"  * Unique Days:       {report['unique_dates']} daily timestamps")
    print(f"  * Invalid Dates:     {report['invalid_dates']} (All dates valid)")
    print("-" * 70)
    print("DATA INTEGRITY & QUALITY CHECKS:")
    print(f"  * Missing Values:    {report['total_missing']} missing values across all columns")
    print(f"  * Duplicate Rows:    {report['duplicate_rows']} identical rows")
    print(f"  * Key Duplicates:    {report['duplicate_series_keys']} duplicate (Store ID + Product ID + Date) keys")
    print(f"  * Records Removed:   0 records deleted (100% record retention)")
    print("-" * 70)
    print("TARGET VARIABLE & LEAKAGE PREVENTION:")
    print(f"  * Target Variable:   '{report['target_column']}' (Intact & Valid)")
    print(f"  * Target Stats:      Min = {report['target_min']}, Max = {report['target_max']}, Mean = {report['target_mean']}")
    if report["dropped_columns"]:
        print(f"  * Excluded Features: {report['dropped_columns']} (Preventing Data Leakage)")
    else:
        print("  * Excluded Features: None")
    print("=" * 70)
    print("STATUS: Cleaned dataset saved successfully!")
    print("=" * 70)


def preprocess_pipeline(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Run end-to-end preprocessing pipeline."""
    input_p = Path(input_path)
    output_p = Path(output_path)

    raw_df = load_raw_data(input_p)
    clean_df, report = validate_and_clean_data(raw_df, drop_leakage=drop_leakage)
    save_processed_data(clean_df, output_p)
    print_summary_report(report, input_p, output_p)
    return clean_df


def main():
    parser = argparse.ArgumentParser(
        description="SmartStorage Data Preprocessing Pipeline (Mahendar)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=f"Path to canonical raw CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to output cleaned CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--keep-leakage",
        action="store_true",
        help="Keep Demand Forecast in dataset (not recommended for model training)",
    )
    args = parser.parse_args()

    preprocess_pipeline(
        input_path=args.input,
        output_path=args.output,
        drop_leakage=not args.keep_leakage,
    )


if __name__ == "__main__":
    main()
