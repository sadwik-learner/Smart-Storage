"""
run_replenishment_demo.py
=========================
End-to-end demonstration of the Smart Store replenishment engine.

What this script does
---------------------
1. Loads the pre-generated operational dataset (73,100 rows, 23 columns).
2. Builds a small synthetic forecast fixture using products and stores that
   actually exist in the dataset. These forecasts are clearly labelled as
   SYNTHETIC / DEMO data -- they are NOT from the team's forecasting model.
3. Selects the most recent inventory record for each product-store pair.
4. Runs the replenishment engine.
5. Saves results to outputs/demo_replenishment_recommendations.csv.
6. Prints a concise summary to the terminal.

Using a real forecast file
--------------------------
If the forecasting team has exported a forecast CSV, pass it on the command
line to replace the synthetic fixtures:

    python scripts/run_replenishment_demo.py \\
        --forecast-csv path/to/real_forecasts.csv

The real forecast CSV must contain at least these columns:
    Product ID, Store ID, forecast_units

where forecast_units is the average predicted daily demand for that
product-store pair.

For daily time-series forecasts (one row per future date), also include a
'forecast_date' column and pass --forecast-mode timeseries.
"""

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the Python path when running this script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.replenishment import build_recommendations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OPERATIONAL_CSV = ROOT / "dataset" / "retail_inventory_with_operations.csv"
OUTPUT_CSV      = ROOT / "outputs" / "demo_replenishment_recommendations.csv"


# ---------------------------------------------------------------------------
# Synthetic forecast fixture builder
# ---------------------------------------------------------------------------

def make_synthetic_forecast_fixture(products: list, stores: list) -> pd.DataFrame:
    """Return a small synthetic forecast DataFrame for demonstration.

    The demand values are plausible placeholder numbers, NOT model predictions.
    This fixture is used only when no real forecast file is provided.

    Parameters
    ----------
    products : list of product IDs that exist in the dataset
    stores   : list of store IDs that exist in the dataset

    Returns
    -------
    DataFrame with columns: Product ID, Store ID, forecast_units
    """
    demand_by_product = {}
    for i, pid in enumerate(sorted(products)):
        # Stagger from ~10 to ~60 units/day to get a variety of recommendation states
        demand_by_product[pid] = round(10 + (i % 10) * 5 + (i // 10) * 3, 1)

    rows = []
    for pid in products:
        for sid in stores:
            rows.append({
                "Product ID": pid,
                "Store ID": sid,
                "forecast_units": demand_by_product[pid],
                "_is_synthetic_fixture": True,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(
    *,
    forecast_csv: "str | None" = None,
    forecast_mode: str = "scalar",
    safety_days: float = 2.0,
    output_path: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    """Load data, build recommendations, save and summarise results."""

    # ------------------------------------------------------------------
    # Load operational data
    # ------------------------------------------------------------------
    if not OPERATIONAL_CSV.exists():
        sys.exit(
            f"[ERROR] Operational dataset not found: {OPERATIONAL_CSV}\n"
            "Run this first:\n"
            "  python -m src.synthetic_operational_data "
            "--input dataset/retail_store_inventory.csv "
            "--output dataset/retail_inventory_with_operations.csv --seed 42"
        )

    print(f"Loading operational data from: {OPERATIONAL_CSV}")
    ops = pd.read_csv(OPERATIONAL_CSV)
    print(f"  -> {len(ops):,} rows, {ops['Product ID'].nunique()} products, "
          f"{ops['Store ID'].nunique()} stores")

    # ------------------------------------------------------------------
    # Prepare inventory snapshot (latest record per store-product)
    # ------------------------------------------------------------------
    ops["Date"] = pd.to_datetime(ops["Date"], errors="coerce")
    inventory_snapshot = (
        ops.sort_values("Date")
        .groupby(["Store ID", "Product ID"], as_index=False)
        .tail(1)
        [["Store ID", "Product ID", "Inventory Level", "Date"]]
    )
    print(f"  -> {len(inventory_snapshot):,} unique product-store pairs "
          f"(latest record per pair)")

    # ------------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------------
    if forecast_csv:
        print(f"\nLoading REAL forecast from: {forecast_csv}")
        forecast_df = pd.read_csv(forecast_csv)
        print(f"  -> {len(forecast_df):,} forecast rows")
    else:
        print("\n[DEMO] No real forecast supplied. Using SYNTHETIC fixture values.")
        print("       These are NOT from the team's model -- for demonstration only.")
        products = ops["Product ID"].unique().tolist()
        stores   = ops["Store ID"].unique().tolist()
        forecast_df = make_synthetic_forecast_fixture(products, stores)
        # Drop internal marker before passing to engine
        forecast_df = forecast_df.drop(columns=["_is_synthetic_fixture"])
        print(f"  -> {len(forecast_df):,} synthetic forecast rows created")

    # ------------------------------------------------------------------
    # Run replenishment engine
    # ------------------------------------------------------------------
    print("\nRunning replenishment engine...")
    recommendations = build_recommendations(
        inventory_df=inventory_snapshot,
        forecast_df=forecast_df,
        operational_df=ops,
        safety_days=safety_days,
        forecast_mode=forecast_mode,
    )

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output_path, index=False)
    print(f"\nSaved {len(recommendations):,} recommendations -> {output_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_total    = len(recommendations)
    n_required = int(recommendations["replenishment_required"].sum())
    n_none     = n_total - n_required
    n_blocked  = int(recommendations["blocked_by_capacity"].sum())
    n_high     = int((recommendations["risk_level"] == "HIGH").sum())
    n_moderate = int((recommendations["risk_level"] == "MODERATE").sum())
    n_low      = int((recommendations["risk_level"] == "LOW").sum())

    print()
    print("=" * 60)
    print("  SMART STORE - REPLENISHMENT SUMMARY (DEMO)")
    print("=" * 60)
    print(f"  Total product-store pairs evaluated : {n_total:>6,}")
    print(f"  Replenishment REQUIRED              : {n_required:>6,}")
    print(f"  No action needed                    : {n_none:>6,}")
    print(f"  Blocked by storage capacity         : {n_blocked:>6,}")
    print()
    print(f"  Risk breakdown:")
    print(f"    HIGH     : {n_high:>6,}")
    print(f"    MODERATE : {n_moderate:>6,}")
    print(f"    LOW      : {n_low:>6,}")
    print("=" * 60)

    if n_required > 0:
        print("\nTop 5 highest-priority recommendations:")
        top5 = (
            recommendations[recommendations["replenishment_required"]]
            .sort_values(["risk_level", "reorder_point"], ascending=[True, False])
            .head(5)
            [["product_id", "store_id", "recommended_order_quantity",
              "estimated_order_cost", "risk_level", "blocked_by_capacity"]]
        )
        print(top5.to_string(index=False))

    print()
    if forecast_csv is None:
        print("NOTE: Forecasts used above are SYNTHETIC demo fixtures.")
        print("      When real model forecasts are ready, run:")
        print("      python scripts/run_replenishment_demo.py "
              "--forecast-csv <path_to_real_forecasts.csv>")

    return recommendations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-csv", default=None,
        help="Path to a real forecast CSV (Product ID, Store ID, forecast_units). "
             "If omitted, synthetic demo fixtures are used."
    )
    parser.add_argument(
        "--forecast-mode", choices=["scalar", "timeseries"], default="scalar",
        help="'scalar' = one average value per product-store (default). "
             "'timeseries' = one row per future date with a 'forecast_date' column."
    )
    parser.add_argument(
        "--safety-days", type=float, default=2.0,
        help="Safety buffer days (default 2.0 for demo -- more conservative than engine default)."
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_CSV),
        help="Output CSV path (default: outputs/demo_replenishment_recommendations.csv)"
    )
    args = parser.parse_args()

    run_demo(
        forecast_csv=args.forecast_csv,
        forecast_mode=args.forecast_mode,
        safety_days=args.safety_days,
        output_path=Path(args.output),
    )
