"""
Replenishment engine for Smart Store.

This module joins current inventory data, demand forecasts, and synthetic
operational data to produce per-product, per-store replenishment recommendations.

Expected input CSV formats
--------------------------

inventory CSV columns (required):
    Product ID   : product identifier matching operational data (e.g. 'P0001')
    Store ID     : store identifier (e.g. 'S001')
    Inventory Level : current on-hand units (integer >= 0)
    Date         : (optional) record date; if present, the latest record per
                   Store ID + Product ID pair is used automatically.

forecast CSV columns (required):
    Product ID   : must match inventory and operational data
    Store ID     : must match inventory data
    forecast_units : average forecast demand per day (float > 0)
                     OR a daily time-series file with one row per future date;
                     see the 'Daily time-series forecasts' section below.

Daily time-series forecasts (advanced)
---------------------------------------
If the forecasting team exports one row per future date, the file should also
contain a 'forecast_date' column.  Pass `forecast_mode='timeseries'` to
`build_recommendations()`.  The engine will select the first `lead_time_days`
rows for each product-store pair and sum them to get lead-time demand, instead
of multiplying a scalar average.

operational CSV columns (required):
    Product ID, supplier_id, lead_time_days, minimum_order_quantity,
    unit_cost, ordering_cost, annual_holding_cost_per_unit,
    storage_capacity_units

output CSV columns
------------------
    product_id, store_id, supplier_id,
    inventory_position, daily_forecast_units, lead_time_days,
    expected_lead_time_demand, safety_stock_units, reorder_point,
    eoq_units, recommended_order_quantity, unit_cost,
    estimated_order_cost, replenishment_required, risk_level,
    blocked_by_capacity, recommendation_reason
"""

import argparse
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

from src.inventory_formulas import (
    SafetyStockPolicy,
    calculate_eoq,
    calculate_inventory_position,
    calculate_reorder_point,
    calculate_safety_stock,
    expected_lead_time_demand,
    recommend_order_quantity,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(name: str, df: pd.DataFrame, required: set) -> None:
    """Raise ValueError listing every missing column."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"The '{name}' DataFrame is missing required columns: {missing}"
        )


def _validate_positive_values(df: pd.DataFrame, column: str, label: str) -> None:
    """Raise ValueError if any value in *column* is not > 0."""
    bad = df[df[column] <= 0]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} rows in '{label}' have non-positive values in '{column}'. "
            f"Example: {bad.iloc[0].to_dict()}"
        )


def _assign_risk_level(row: pd.Series) -> str:
    """Return 'HIGH', 'MODERATE', or 'LOW' based on inventory status.

    HIGH     : blocked by capacity, OR inventory position is already below
               expected lead-time demand (stockout very likely before reorder arrives).
    MODERATE : replenishment required but position is still above lead-time demand.
    LOW      : no replenishment required.
    """
    if row["blocked_by_capacity"]:
        return "HIGH"
    if not row["replenishment_required"]:
        return "LOW"
    if row["inventory_position"] < row["expected_lead_time_demand"]:
        return "HIGH"
    return "MODERATE"


# ---------------------------------------------------------------------------
# Main engine function
# ---------------------------------------------------------------------------

def build_recommendations(
    inventory_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    operational_df: pd.DataFrame,
    *,
    forecast_column: str = "forecast_units",
    forecast_mode: Literal["scalar", "timeseries"] = "scalar",
    safety_days: float = 1.0,
    safety_stock_policy: SafetyStockPolicy = SafetyStockPolicy.FIXED_DAYS,
) -> pd.DataFrame:
    """Build a replenishment recommendation for every product-store pair.

    Parameters
    ----------
    inventory_df    : DataFrame with current inventory records.
    forecast_df     : DataFrame with demand forecasts.
    operational_df  : DataFrame with synthetic supplier / cost fields.
    forecast_column : name of the forecast demand column (default 'forecast_units').
    forecast_mode   : 'scalar' (one average value per product-store) or
                      'timeseries' (one row per future date; requires
                      'forecast_date' column in forecast_df).
    safety_days     : number of extra demand days to hold as safety stock
                      (used when safety_stock_policy is FIXED_DAYS).
    safety_stock_policy : SafetyStockPolicy enum value.

    Returns
    -------
    DataFrame with one row per product-store pair and recommendation columns.

    Raises
    ------
    ValueError : on missing columns, invalid values, or missing forecasts.
    """

    # ------------------------------------------------------------------
    # 1. Validate input columns
    # ------------------------------------------------------------------
    _validate_columns("inventory", inventory_df, {"Product ID", "Store ID", "Inventory Level"})
    _validate_columns("forecasts", forecast_df, {"Product ID", "Store ID", forecast_column})
    _validate_columns(
        "operations", operational_df,
        {
            "Product ID", "supplier_id", "lead_time_days", "minimum_order_quantity",
            "unit_cost", "ordering_cost", "annual_holding_cost_per_unit",
            "storage_capacity_units",
        },
    )

    # ------------------------------------------------------------------
    # 2. Validate value ranges
    # ------------------------------------------------------------------
    inv = inventory_df.copy()
    if (inv["Inventory Level"] < 0).any():
        raise ValueError("'Inventory Level' contains negative values.")

    _validate_positive_values(operational_df, "lead_time_days", "operations")
    _validate_positive_values(operational_df, "minimum_order_quantity", "operations")
    _validate_positive_values(operational_df, "unit_cost", "operations")
    _validate_positive_values(operational_df, "ordering_cost", "operations")
    _validate_positive_values(operational_df, "annual_holding_cost_per_unit", "operations")
    _validate_positive_values(operational_df, "storage_capacity_units", "operations")

    if (forecast_df[forecast_column] < 0).any():
        raise ValueError(f"'{forecast_column}' contains negative values.")

    # ------------------------------------------------------------------
    # 3. Select latest inventory record per store-product pair
    # ------------------------------------------------------------------
    if "Date" in inv.columns:
        inv["Date"] = pd.to_datetime(inv["Date"], errors="coerce")
        inv = inv.sort_values("Date").groupby(
            ["Store ID", "Product ID"], as_index=False
        ).tail(1)

    inv = inv.drop_duplicates(["Store ID", "Product ID"], keep="last")

    # ------------------------------------------------------------------
    # 4. Prepare forecast data
    # ------------------------------------------------------------------
    if forecast_mode == "timeseries":
        _validate_columns("forecasts (timeseries)", forecast_df, {"forecast_date"})
        # For timeseries mode: store per-product-store list of sorted daily values
        fc_grouped = (
            forecast_df
            .sort_values("forecast_date")
            .groupby(["Store ID", "Product ID"])[forecast_column]
            .apply(list)
            .reset_index()
            .rename(columns={forecast_column: "_fc_series"})
        )
        # Also keep a scalar average for reporting
        fc_avg = (
            forecast_df
            .groupby(["Store ID", "Product ID"])[forecast_column]
            .mean()
            .reset_index()
            .rename(columns={forecast_column: "_fc_scalar"})
        )
        fc_data = fc_grouped.merge(fc_avg, on=["Store ID", "Product ID"])
    else:
        # scalar mode: one average value per product-store
        fc_data = forecast_df[["Store ID", "Product ID", forecast_column]].copy()

    # ------------------------------------------------------------------
    # 5. Deduplicate operational data (one row per product, op columns only)
    # ------------------------------------------------------------------
    op_cols = [
        "Product ID", "supplier_id", "lead_time_days", "minimum_order_quantity",
        "unit_cost", "ordering_cost", "annual_holding_cost_per_unit",
        "storage_capacity_units",
    ]
    ops = operational_df[op_cols].drop_duplicates("Product ID")

    # ------------------------------------------------------------------
    # 6. Join all data
    # ------------------------------------------------------------------
    merged = inv.merge(
        fc_data, on=["Store ID", "Product ID"], how="left", validate="one_to_one"
    ).merge(
        ops, on="Product ID", how="left", validate="many_to_one"
    )

    # ------------------------------------------------------------------
    # 7. Detect and report missing forecasts
    # ------------------------------------------------------------------
    if forecast_mode == "timeseries":
        fc_missing_mask = merged["_fc_series"].isna()
    else:
        fc_missing_mask = merged[forecast_column].isna()

    if fc_missing_mask.any():
        missing_pairs = merged.loc[fc_missing_mask, ["Store ID", "Product ID"]].to_dict("records")
        raise ValueError(
            f"{fc_missing_mask.sum()} product-store pair(s) have no forecast. "
            f"No values were invented. Missing pairs: {missing_pairs[:5]}"
            + (" (and more...)" if len(missing_pairs) > 5 else "")
        )

    # ------------------------------------------------------------------
    # 8. Calculate recommendations row by row
    # ------------------------------------------------------------------
    results = []

    for _, row in merged.iterrows():
        lead = int(row["lead_time_days"])
        on_hand = float(row["Inventory Level"])

        # --- 8a. Forecast demand ---
        if forecast_mode == "timeseries":
            daily_scalar = float(row["_fc_scalar"])
            daily_series = row["_fc_series"]
            if len(daily_series) < lead:
                warnings.warn(
                    f"Timeseries forecast for {row['Product ID']} / {row['Store ID']} "
                    f"has only {len(daily_series)} days but lead time is {lead} days. "
                    f"Falling back to scalar average.",
                    UserWarning,
                    stacklevel=2,
                )
                ltd = expected_lead_time_demand(daily_scalar, lead)
            else:
                ltd = expected_lead_time_demand(daily_series, lead)
        else:
            daily_scalar = float(row[forecast_column])
            ltd = expected_lead_time_demand(daily_scalar, lead)

        # --- 8b. Safety stock ---
        safety = calculate_safety_stock(
            daily_scalar,
            policy=safety_stock_policy,
            safety_days=safety_days,
            lead_time_days=lead,
        )

        # --- 8c. Reorder point ---
        rop = calculate_reorder_point(ltd, safety)

        # --- 8d. Inventory position ---
        # on_order and backorders are not in the base dataset; we use 0 and
        # note this in the recommendation so the user knows what is assumed.
        position = calculate_inventory_position(on_hand)
        position_note = "on-hand only (on-order and backorders not available in this dataset)"

        # --- 8e. Annual demand estimate and EOQ ---
        annual_demand = daily_scalar * 365
        eoq = calculate_eoq(
            annual_demand,
            float(row["ordering_cost"]),
            float(row["annual_holding_cost_per_unit"]),
        )

        # --- 8f. Recommendation ---
        rec = recommend_order_quantity(
            inventory_position=position,
            reorder_point=rop,
            eoq_units=eoq,
            minimum_order_quantity=int(row["minimum_order_quantity"]),
            storage_capacity_units=int(row["storage_capacity_units"]),
            on_hand_units=on_hand,
        )

        qty = rec["recommended_order_quantity"]
        estimated_cost = round(qty * float(row["unit_cost"]), 2)

        result_row = {
            "product_id": row["Product ID"],
            "store_id": row["Store ID"],
            "supplier_id": row["supplier_id"],
            "on_hand_units": on_hand,
            "inventory_position": position,
            "inventory_position_note": position_note,
            "daily_forecast_units": round(daily_scalar, 4),
            "lead_time_days": lead,
            "expected_lead_time_demand": round(ltd, 2),
            "safety_stock_units": safety,
            "reorder_point": rop,
            "eoq_units": eoq,
            "minimum_order_quantity": int(row["minimum_order_quantity"]),
            "storage_capacity_units": int(row["storage_capacity_units"]),
            "recommended_order_quantity": qty,
            "unit_cost": float(row["unit_cost"]),
            "estimated_order_cost": estimated_cost,
            "replenishment_required": rec["replenishment_required"],
            "blocked_by_capacity": rec["blocked_by_capacity"],
            "recommendation_reason": rec["recommendation_reason"],
        }

        # Add risk level (needs replenishment_required, blocked_by_capacity, inventory_position,
        # expected_lead_time_demand)
        result_row["risk_level"] = _assign_risk_level(
            pd.Series(result_row)
        )

        results.append(result_row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",   required=True, help="Path to inventory CSV")
    p.add_argument("--forecasts",   required=True, help="Path to forecast CSV")
    p.add_argument("--operations",  required=True, help="Path to operational data CSV")
    p.add_argument("--output",      required=True, help="Path to write recommendations CSV")
    p.add_argument("--safety-days", type=float, default=1.0,
                   help="Extra safety buffer days (default 1.0)")
    p.add_argument("--forecast-mode", choices=["scalar", "timeseries"], default="scalar",
                   help="Forecast format: 'scalar' (one value per product-store) "
                        "or 'timeseries' (one row per future date with 'forecast_date' column)")
    a = p.parse_args()

    result = build_recommendations(
        pd.read_csv(a.inventory),
        pd.read_csv(a.forecasts),
        pd.read_csv(a.operations),
        safety_days=a.safety_days,
        forecast_mode=a.forecast_mode,
    )

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
    n_yes = int(result["replenishment_required"].sum())
    n_blocked = int(result["blocked_by_capacity"].sum())
    print(f"Saved {len(result)} recommendations to {a.output}")
    print(f"  Replenishment required : {n_yes}")
    print(f"  No action needed       : {len(result) - n_yes}")
    print(f"  Blocked by capacity    : {n_blocked}")


if __name__ == "__main__":
    main()
