"""Generate reproducible synthetic supplier/operational fields for Smart Store."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_operational_data(df: pd.DataFrame, seed: int = 42, supplier_count: int = 8) -> pd.DataFrame:
    if "Product ID" not in df.columns:
        raise ValueError("Input data must contain 'Product ID'")
    if supplier_count < 1:
        raise ValueError("supplier_count must be at least 1")
    rng = np.random.default_rng(seed)
    products = df[["Product ID"]].drop_duplicates().reset_index(drop=True)
    n = len(products)
    products["supplier_id"] = [f"SUP{i:03d}" for i in rng.integers(1, supplier_count + 1, n)]
    products["lead_time_days"] = rng.integers(2, 11, n)
    products["minimum_order_quantity"] = rng.integers(10, 101, n)
    products["unit_cost"] = np.round(rng.uniform(5, 500, n), 2)
    products["ordering_cost"] = np.round(rng.uniform(50, 250, n), 2)
    products["annual_holding_cost_per_unit"] = np.round(products["unit_cost"] * rng.uniform(.15, .30, n), 2)
    products["storage_capacity_units"] = rng.integers(100, 2001, n)
    products["operational_data_is_synthetic"] = True
    generated = [c for c in products.columns if c != "Product ID"]
    clean = df.drop(columns=[c for c in generated if c in df.columns], errors="ignore")
    return clean.merge(products, on="Product ID", how="left", validate="many_to_one")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(args.input)
    df = pd.read_csv(args.input)
    result = generate_operational_data(df, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} rows to {args.output}. Supplier/cost fields are synthetic assumptions.")

if __name__ == "__main__":
    main()
