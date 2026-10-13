"""
Comprehensive test suite for Smart Store inventory engine.

Covers:
  - All inventory formula functions (safety stock, EOQ, ROP, position, recommendations)
  - Replenishment engine (normal, edge, and error cases)
  - Operational data generation (reproducibility, column presence, validation)

Run with:
    python -m unittest discover -s tests -v
"""

import unittest
import warnings
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.inventory_formulas import (
    SafetyStockPolicy,
    calculate_safety_stock,
    expected_lead_time_demand,
    calculate_reorder_point,
    calculate_eoq,
    calculate_inventory_position,
    recommend_order_quantity,
)
from src.replenishment import build_recommendations
from src.synthetic_operational_data import generate_operational_data


# ===========================================================================
# Helper factories
# ===========================================================================

def _make_inventory(rows):
    """Create a minimal inventory DataFrame from a list of dicts."""
    return pd.DataFrame(rows)


def _make_forecast(rows):
    return pd.DataFrame(rows)


def _make_ops(rows):
    return pd.DataFrame(rows)


def _base_ops(product_id="P001"):
    """Return a single-row operational DataFrame with sensible defaults."""
    return _make_ops([{
        "Product ID": product_id,
        "supplier_id": "SUP001",
        "lead_time_days": 5,
        "minimum_order_quantity": 20,
        "unit_cost": 100.0,
        "ordering_cost": 150.0,
        "annual_holding_cost_per_unit": 20.0,
        "storage_capacity_units": 1000,
    }])


def _base_inv(product_id="P001", store_id="S001", level=500):
    return _make_inventory([{
        "Product ID": product_id,
        "Store ID": store_id,
        "Inventory Level": level,
    }])


def _base_fc(product_id="P001", store_id="S001", units=30.0):
    return _make_forecast([{
        "Product ID": product_id,
        "Store ID": store_id,
        "forecast_units": units,
    }])


# ===========================================================================
# 1. Safety stock tests
# ===========================================================================

class TestSafetyStock(unittest.TestCase):

    def test_fixed_days_basic(self):
        """20 units/day × 1 safety day → 20."""
        self.assertEqual(calculate_safety_stock(20, safety_days=1), 20)

    def test_fixed_days_rounds_up(self):
        """7.5 × 2 = 15 → 15 (no rounding needed, check fractional demand)."""
        self.assertEqual(calculate_safety_stock(7.3, safety_days=2), 15)

    def test_fixed_days_default_one_day(self):
        """Default safety_days=1; result should equal ceil(demand)."""
        self.assertEqual(calculate_safety_stock(15.5), 16)

    def test_z_score_policy_zero_sigma(self):
        """With σ_demand=0 and σ_lead=0, safety stock should be 0."""
        result = calculate_safety_stock(
            20,
            policy=SafetyStockPolicy.Z_SCORE,
            z_score=1.65,
            sigma_demand=0,
            lead_time_days=5,
            sigma_lead_time=0,
        )
        self.assertEqual(result, 0)

    def test_z_score_policy_nonzero_sigma(self):
        """With demand variability, safety stock must be > 0."""
        result = calculate_safety_stock(
            20,
            policy=SafetyStockPolicy.Z_SCORE,
            z_score=1.65,
            sigma_demand=5,
            lead_time_days=3,
            sigma_lead_time=0,
        )
        self.assertGreater(result, 0)

    def test_negative_demand_raises(self):
        with self.assertRaises(ValueError):
            calculate_safety_stock(-5)

    def test_zero_safety_days_raises(self):
        with self.assertRaises(ValueError):
            calculate_safety_stock(10, safety_days=0)

    def test_invalid_policy_raises(self):
        with self.assertRaises(ValueError):
            calculate_safety_stock(10, policy="bogus")  # type: ignore


# ===========================================================================
# 2. Expected lead-time demand tests
# ===========================================================================

class TestExpectedLeadTimeDemand(unittest.TestCase):

    def test_scalar_mode(self):
        """20 units/day × 3 days → 60."""
        self.assertEqual(expected_lead_time_demand(20, 3), 60)

    def test_sequence_mode_exact_length(self):
        """[10, 12, 14] → 36 (not multiplied by lead time again)."""
        self.assertEqual(expected_lead_time_demand([10, 12, 14], 3), 36)

    def test_sequence_mode_longer_than_lead_time(self):
        """Sequence longer than lead time: only first lead_time values summed."""
        self.assertEqual(expected_lead_time_demand([10, 20, 30, 40], 3), 60)

    def test_sequence_too_short_raises(self):
        with self.assertRaises(ValueError):
            expected_lead_time_demand([10, 20], 5)

    def test_negative_scalar_raises(self):
        with self.assertRaises(ValueError):
            expected_lead_time_demand(-5, 3)

    def test_negative_value_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            expected_lead_time_demand([10, -1, 15], 3)

    def test_zero_lead_time_raises(self):
        with self.assertRaises(ValueError):
            expected_lead_time_demand(10, 0)

    def test_zero_demand_allowed(self):
        """Zero demand is valid (e.g. seasonal product)."""
        self.assertEqual(expected_lead_time_demand(0, 3), 0)


# ===========================================================================
# 3. Reorder point tests
# ===========================================================================

class TestReorderPoint(unittest.TestCase):

    def test_basic(self):
        """Lead-time demand 60 + safety stock 15 → ROP 75."""
        self.assertEqual(calculate_reorder_point(60, 15), 75)

    def test_rounds_up(self):
        self.assertEqual(calculate_reorder_point(10.1, 5.2), 16)

    def test_negative_safety_stock_raises(self):
        with self.assertRaises(ValueError):
            calculate_reorder_point(60, -1)

    def test_negative_ltd_raises(self):
        with self.assertRaises(ValueError):
            calculate_reorder_point(-10, 5)


# ===========================================================================
# 4. EOQ tests
# ===========================================================================

class TestEOQ(unittest.TestCase):

    def test_basic(self):
        """sqrt(2 × 7300 × 100 / 20) ≈ 270.19 → ceil = 271."""
        self.assertEqual(calculate_eoq(7300, 100, 20), 271)

    def test_zero_demand_returns_zero(self):
        self.assertEqual(calculate_eoq(0, 100, 20), 0)

    def test_zero_holding_cost_raises(self):
        with self.assertRaises(ValueError):
            calculate_eoq(1000, 50, 0)

    def test_negative_ordering_cost_raises(self):
        with self.assertRaises(ValueError):
            calculate_eoq(1000, -50, 20)

    def test_negative_annual_demand_raises(self):
        with self.assertRaises(ValueError):
            calculate_eoq(-100, 50, 20)

    def test_result_is_integer(self):
        result = calculate_eoq(5000, 200, 15)
        self.assertIsInstance(result, int)


# ===========================================================================
# 5. Inventory position tests
# ===========================================================================

class TestInventoryPosition(unittest.TestCase):

    def test_basic(self):
        """on_hand=100, on_order=20, backorders=5 → 115."""
        self.assertEqual(calculate_inventory_position(100, 20, 5), 115)

    def test_defaults_to_on_hand(self):
        self.assertEqual(calculate_inventory_position(200), 200)

    def test_negative_on_hand_raises(self):
        with self.assertRaises(ValueError):
            calculate_inventory_position(-10)

    def test_negative_on_order_raises(self):
        with self.assertRaises(ValueError):
            calculate_inventory_position(100, -5)

    def test_zero_values_allowed(self):
        self.assertEqual(calculate_inventory_position(0, 0, 0), 0)


# ===========================================================================
# 6. Recommend order quantity tests
# ===========================================================================

class TestRecommendOrderQuantity(unittest.TestCase):

    # --- above ROP (no order needed) ---
    def test_no_order_above_rop(self):
        r = recommend_order_quantity(100, 75, 271, 20)
        self.assertFalse(r["replenishment_required"])
        self.assertEqual(r["recommended_order_quantity"], 0)
        self.assertFalse(r["blocked_by_capacity"])

    # --- at ROP (order needed) ---
    def test_order_at_rop(self):
        r = recommend_order_quantity(75, 75, 100, 20)
        self.assertTrue(r["replenishment_required"])
        self.assertGreater(r["recommended_order_quantity"], 0)

    # --- below ROP (order needed) ---
    def test_order_below_rop(self):
        r = recommend_order_quantity(50, 75, 100, 20)
        self.assertTrue(r["replenishment_required"])
        self.assertGreater(r["recommended_order_quantity"], 0)

    # --- MOQ constraint ---
    def test_moq_applied_when_eoq_smaller(self):
        """EOQ=10 but MOQ=20 → recommended qty should be 20."""
        r = recommend_order_quantity(50, 75, 10, 20)
        self.assertEqual(r["recommended_order_quantity"], 20)

    def test_eoq_applied_when_moq_smaller(self):
        """EOQ=100, MOQ=20 → recommended qty should be 100."""
        r = recommend_order_quantity(50, 75, 100, 20)
        self.assertEqual(r["recommended_order_quantity"], 100)

    # --- capacity blocked ---
    def test_capacity_below_moq_is_blocked(self):
        """Remaining room (10) < MOQ (20) → blocked."""
        r = recommend_order_quantity(50, 75, 100, 20,
                                     storage_capacity_units=110,
                                     on_hand_units=100)
        self.assertTrue(r["blocked_by_capacity"])
        self.assertEqual(r["recommended_order_quantity"], 0)

    # --- capacity limits but does not block ---
    def test_capacity_limits_quantity(self):
        """Desired=100, room=50, MOQ=20 → qty should be 50."""
        r = recommend_order_quantity(50, 75, 100, 20,
                                     storage_capacity_units=150,
                                     on_hand_units=100)
        self.assertFalse(r["blocked_by_capacity"])
        self.assertEqual(r["recommended_order_quantity"], 50)

    # --- missing on_hand_units ---
    def test_missing_on_hand_with_capacity_raises(self):
        with self.assertRaises(ValueError):
            recommend_order_quantity(50, 75, 100, 20,
                                     storage_capacity_units=500)

    # --- invalid inputs ---
    def test_negative_eoq_raises(self):
        with self.assertRaises(ValueError):
            recommend_order_quantity(50, 75, -10, 20)

    def test_zero_moq_raises(self):
        with self.assertRaises(ValueError):
            recommend_order_quantity(50, 75, 100, 0)

    # --- human approval wording must be in reason ---
    def test_approval_required_message(self):
        r = recommend_order_quantity(50, 75, 100, 20)
        self.assertIn("approval", r["recommendation_reason"].lower())


# ===========================================================================
# 7. Replenishment engine tests
# ===========================================================================

class TestBuildRecommendations(unittest.TestCase):

    def _run(self, inv_level=500, forecast=30.0, lead=5, moq=20,
             capacity=1000, on_hand=None):
        """Convenience wrapper: run engine with minimal defaults."""
        inv = _base_inv(level=inv_level)
        fc  = _base_fc(units=forecast)
        ops = _make_ops([{
            "Product ID": "P001",
            "supplier_id": "SUP001",
            "lead_time_days": lead,
            "minimum_order_quantity": moq,
            "unit_cost": 100.0,
            "ordering_cost": 150.0,
            "annual_holding_cost_per_unit": 20.0,
            "storage_capacity_units": capacity,
        }])
        return build_recommendations(inv, fc, ops, safety_days=1.0)

    # --- normal data ---
    def test_returns_dataframe(self):
        result = self._run()
        self.assertIsInstance(result, pd.DataFrame)

    def test_output_columns_present(self):
        result = self._run()
        required_cols = [
            "product_id", "store_id", "supplier_id",
            "inventory_position", "daily_forecast_units",
            "lead_time_days", "expected_lead_time_demand",
            "safety_stock_units", "reorder_point",
            "eoq_units", "recommended_order_quantity",
            "unit_cost", "estimated_order_cost",
            "replenishment_required", "risk_level",
            "blocked_by_capacity", "recommendation_reason",
        ]
        for col in required_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_one_row_per_product_store(self):
        result = self._run()
        self.assertEqual(len(result), 1)

    # --- above ROP ---
    def test_above_rop_no_replenishment(self):
        """Inventory=1000, daily_forecast=10, lead=3, safety_days=1
           → ROP = 30 + 10 = 40; position 1000 > 40 → no replenishment."""
        result = self._run(inv_level=1000, forecast=10, lead=3, capacity=5000)
        self.assertFalse(result.iloc[0]["replenishment_required"])
        self.assertEqual(result.iloc[0]["recommended_order_quantity"], 0)
        self.assertEqual(result.iloc[0]["risk_level"], "LOW")

    # --- at ROP ---
    def test_at_rop_triggers_replenishment(self):
        """Position == ROP should trigger an order."""
        # ROP = forecast * lead + safety = 10 * 3 + 10 = 40; set inv=40
        result = self._run(inv_level=40, forecast=10, lead=3, capacity=5000)
        self.assertTrue(result.iloc[0]["replenishment_required"])

    # --- below ROP ---
    def test_below_rop_triggers_replenishment(self):
        result = self._run(inv_level=5, forecast=30, lead=5, capacity=5000)
        self.assertTrue(result.iloc[0]["replenishment_required"])
        self.assertGreater(result.iloc[0]["recommended_order_quantity"], 0)

    # --- MOQ constraint ---
    def test_moq_respected(self):
        """When EOQ < MOQ, recommended quantity must be >= MOQ.

        Use inv_level=0 to guarantee position (0) <= ROP, so an order is
        always triggered regardless of forecast size.  EOQ will be small
        because daily_forecast is low, so the engine must round up to MOQ.
        """
        result = self._run(inv_level=0, forecast=0.5, lead=2, moq=50, capacity=5000)
        row = result.iloc[0]
        # The order must actually have been triggered
        self.assertTrue(row["replenishment_required"],
                        "Order should have been triggered at zero inventory")
        qty = row["recommended_order_quantity"]
        moq = row["minimum_order_quantity"]
        self.assertGreaterEqual(qty, moq)

    # --- capacity blocked ---
    def test_capacity_blocks_order(self):
        """When remaining space < MOQ, order must be blocked."""
        # on_hand=990, capacity=1000 → room=10 < MOQ=20 → blocked
        result = self._run(inv_level=990, forecast=200, lead=5, moq=20, capacity=1000)
        self.assertTrue(result.iloc[0]["blocked_by_capacity"])
        self.assertEqual(result.iloc[0]["recommended_order_quantity"], 0)
        self.assertEqual(result.iloc[0]["risk_level"], "HIGH")

    # --- estimated order cost ---
    def test_estimated_cost_calculation(self):
        """estimated_order_cost must equal qty × unit_cost."""
        result = self._run(inv_level=5, forecast=30, lead=5, capacity=5000)
        row = result.iloc[0]
        expected = round(row["recommended_order_quantity"] * row["unit_cost"], 2)
        self.assertAlmostEqual(row["estimated_order_cost"], expected, places=2)

    # --- missing forecast raises ---
    def test_missing_forecast_raises(self):
        inv = _base_inv(product_id="P001")
        fc  = _base_fc(product_id="P999")    # wrong product ID → no match
        ops = _base_ops(product_id="P001")
        with self.assertRaises(ValueError, msg="Should raise for missing forecast"):
            build_recommendations(inv, fc, ops)

    # --- zero annual demand (forecast=0) ---
    def test_zero_forecast_eoq_zero(self):
        """Zero daily forecast → EOQ=0 and position likely above ROP=0."""
        inv = _base_inv(level=100)
        fc  = _base_fc(units=0)
        ops = _base_ops()
        result = build_recommendations(inv, fc, ops)
        self.assertEqual(result.iloc[0]["eoq_units"], 0)

    # --- invalid inventory level ---
    def test_negative_inventory_raises(self):
        inv = _base_inv(level=-5)
        fc  = _base_fc()
        ops = _base_ops()
        with self.assertRaises(ValueError):
            build_recommendations(inv, fc, ops)

    # --- missing columns ---
    def test_missing_inventory_column_raises(self):
        inv = pd.DataFrame([{"Product ID": "P001", "Store ID": "S001"}])  # no Inventory Level
        fc  = _base_fc()
        ops = _base_ops()
        with self.assertRaises(ValueError):
            build_recommendations(inv, fc, ops)

    def test_missing_forecast_column_raises(self):
        inv = _base_inv()
        fc  = pd.DataFrame([{"Product ID": "P001", "Store ID": "S001"}])  # no forecast_units
        ops = _base_ops()
        with self.assertRaises(ValueError):
            build_recommendations(inv, fc, ops)

    # --- multiple products and stores ---
    def test_multiple_products_and_stores(self):
        products = ["P001", "P002", "P003"]
        stores   = ["S001", "S002"]
        inv_rows = [{"Product ID": p, "Store ID": s, "Inventory Level": 100}
                    for p in products for s in stores]
        fc_rows  = [{"Product ID": p, "Store ID": s, "forecast_units": 20.0}
                    for p in products for s in stores]
        ops_rows = [{
            "Product ID": p, "supplier_id": "SUP001",
            "lead_time_days": 3, "minimum_order_quantity": 10,
            "unit_cost": 50.0, "ordering_cost": 100.0,
            "annual_holding_cost_per_unit": 10.0, "storage_capacity_units": 500,
        } for p in products]

        result = build_recommendations(
            _make_inventory(inv_rows),
            _make_forecast(fc_rows),
            _make_ops(ops_rows),
        )
        self.assertEqual(len(result), len(products) * len(stores))

    # --- latest date selected ---
    def test_latest_date_record_selected(self):
        """When multiple dated records exist, the most recent one is used."""
        inv = pd.DataFrame([
            {"Product ID": "P001", "Store ID": "S001", "Inventory Level": 999,
             "Date": "2022-01-01"},
            {"Product ID": "P001", "Store ID": "S001", "Inventory Level": 42,
             "Date": "2024-01-01"},  # latest → should be used
        ])
        fc  = _base_fc()
        ops = _base_ops()
        result = build_recommendations(inv, fc, ops)
        self.assertAlmostEqual(result.iloc[0]["on_hand_units"], 42)

    # --- recommendation count sanity ---
    def test_recommendation_count_matches_pairs(self):
        """Result rows must equal the number of unique product-store pairs."""
        inv = pd.DataFrame([
            {"Product ID": "P001", "Store ID": "S001", "Inventory Level": 50},
            {"Product ID": "P001", "Store ID": "S002", "Inventory Level": 60},
        ])
        fc = pd.DataFrame([
            {"Product ID": "P001", "Store ID": "S001", "forecast_units": 10.0},
            {"Product ID": "P001", "Store ID": "S002", "forecast_units": 10.0},
        ])
        ops = _base_ops("P001")
        result = build_recommendations(inv, fc, ops)
        self.assertEqual(len(result), 2)

    # --- missing on-order data note ---
    def test_position_note_mentions_on_order(self):
        """Recommendation note should acknowledge on-order data is missing."""
        result = self._run()
        note = result.iloc[0]["inventory_position_note"]
        self.assertIn("on-order", note.lower())

    # --- regression: operational_df containing extra columns (like full dataset) ---
    def test_operational_df_with_extra_columns_no_clash(self):
        """If operational_df contains Store ID, Inventory Level etc., merge should not crash."""
        inv = _base_inv(level=50)
        fc  = _base_fc(units=20.0)
        # Ops has columns that also exist in inv:
        ops = pd.DataFrame([{
            "Product ID": "P001",
            "Store ID": "S001",
            "Inventory Level": 9999,  # Should not overwrite or clash with inv
            "supplier_id": "SUP001",
            "lead_time_days": 3,
            "minimum_order_quantity": 10,
            "unit_cost": 50.0,
            "ordering_cost": 100.0,
            "annual_holding_cost_per_unit": 10.0,
            "storage_capacity_units": 500,
        }])
        result = build_recommendations(inv, fc, ops)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["on_hand_units"], 50)

    # --- timeseries forecast mode ---
    def test_timeseries_forecast_mode(self):
        """Timeseries mode should sum daily forecasts over the lead time horizon."""
        inv = _base_inv(level=20)
        # 4 days of forecasts, lead time = 3 -> sum of first 3 = 10 + 15 + 20 = 45
        fc = pd.DataFrame([
            {"Product ID": "P001", "Store ID": "S001", "forecast_date": "2024-01-01", "forecast_units": 10.0},
            {"Product ID": "P001", "Store ID": "S001", "forecast_date": "2024-01-02", "forecast_units": 15.0},
            {"Product ID": "P001", "Store ID": "S001", "forecast_date": "2024-01-03", "forecast_units": 20.0},
            {"Product ID": "P001", "Store ID": "S001", "forecast_date": "2024-01-04", "forecast_units": 25.0},
        ])
        ops = _make_ops([{
            "Product ID": "P001",
            "supplier_id": "SUP001",
            "lead_time_days": 3,
            "minimum_order_quantity": 10,
            "unit_cost": 50.0,
            "ordering_cost": 100.0,
            "annual_holding_cost_per_unit": 10.0,
            "storage_capacity_units": 500,
        }])
        result = build_recommendations(inv, fc, ops, forecast_mode="timeseries")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["expected_lead_time_demand"], 45.0)


# ===========================================================================
# 8. Synthetic operational data tests
# ===========================================================================

class TestSyntheticOperationalData(unittest.TestCase):

    def _make_base_df(self):
        return pd.DataFrame([
            {"Product ID": "P001", "Store ID": "S001", "Inventory Level": 100},
            {"Product ID": "P002", "Store ID": "S001", "Inventory Level": 200},
            {"Product ID": "P001", "Store ID": "S002", "Inventory Level": 150},
        ])

    def test_all_required_columns_present(self):
        df = self._make_base_df()
        result = generate_operational_data(df, seed=42)
        for col in ["supplier_id", "lead_time_days", "minimum_order_quantity",
                    "unit_cost", "ordering_cost", "annual_holding_cost_per_unit",
                    "storage_capacity_units", "operational_data_is_synthetic"]:
            self.assertIn(col, result.columns)

    def test_row_count_preserved(self):
        df = self._make_base_df()
        result = generate_operational_data(df, seed=42)
        self.assertEqual(len(result), len(df))

    def test_reproducibility_same_seed(self):
        df = self._make_base_df()
        r1 = generate_operational_data(df, seed=7)
        r2 = generate_operational_data(df, seed=7)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_differ(self):
        df = self._make_base_df()
        r1 = generate_operational_data(df, seed=1)
        r2 = generate_operational_data(df, seed=2)
        # At least one numeric column should differ
        self.assertFalse((r1["unit_cost"] == r2["unit_cost"]).all())

    def test_supplier_consistent_per_product(self):
        """Same product must get the same supplier across all stores."""
        df = self._make_base_df()
        result = generate_operational_data(df, seed=42)
        self.assertEqual(result[result["Product ID"] == "P001"]["supplier_id"].nunique(), 1)

    def test_lead_times_consistent_per_product(self):
        df = self._make_base_df()
        result = generate_operational_data(df, seed=42)
        self.assertEqual(result[result["Product ID"] == "P001"]["lead_time_days"].nunique(), 1)

    def test_no_missing_values(self):
        df = self._make_base_df()
        result = generate_operational_data(df, seed=42)
        new_cols = ["supplier_id", "lead_time_days", "minimum_order_quantity",
                    "unit_cost", "ordering_cost", "annual_holding_cost_per_unit",
                    "storage_capacity_units"]
        self.assertEqual(result[new_cols].isna().sum().sum(), 0)

    def test_positive_values(self):
        df = self._make_base_df()
        result = generate_operational_data(df, seed=42)
        for col in ["lead_time_days", "minimum_order_quantity", "unit_cost",
                    "ordering_cost", "annual_holding_cost_per_unit",
                    "storage_capacity_units"]:
            self.assertTrue((result[col] > 0).all(), f"{col} has non-positive values")

    def test_missing_product_id_raises(self):
        df = pd.DataFrame([{"Store ID": "S001", "Inventory Level": 100}])
        with self.assertRaises(ValueError):
            generate_operational_data(df)

    def test_invalid_supplier_count_raises(self):
        df = self._make_base_df()
        with self.assertRaises(ValueError):
            generate_operational_data(df, supplier_count=0)

    def test_no_duplicate_columns_on_rerun(self):
        """Re-running on already-enriched data must not create duplicate columns."""
        df = self._make_base_df()
        first  = generate_operational_data(df, seed=42)
        second = generate_operational_data(first, seed=42)
        # No column should appear more than once
        self.assertEqual(len(second.columns), len(set(second.columns)))

    def test_synthetic_flag_is_true(self):
        df = self._make_base_df()
        result = generate_operational_data(df, seed=42)
        self.assertTrue(result["operational_data_is_synthetic"].all())


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
