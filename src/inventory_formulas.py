"""
Inventory formula library for Smart Store.

This module contains pure-Python functions for every inventory calculation
used by the replenishment engine.  All functions are unit-tested in
tests/test_inventory_formulas.py.

Terminology
-----------
- Lead time : number of days between placing an order and receiving it.
- Safety stock : extra buffer units held to cover forecast errors or delays.
- ROP : Reorder Point — inventory position at which we place a new order.
- EOQ : Economic Order Quantity — mathematically optimal order size.
- Inventory position : on-hand + on-order − backorders (the full picture of stock).
"""

import math
from collections.abc import Sequence
from enum import Enum


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check(name: str, value, positive: bool = False) -> None:
    """Raise ValueError if *value* is None, non-finite, or violates the sign rule.

    Parameters
    ----------
    name     : human-readable field name shown in the error message.
    value    : the number to validate.
    positive : if True the value must be strictly > 0; otherwise it must be >= 0.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number, got {value!r}")
    if not math.isfinite(v):
        raise ValueError(f"'{name}' must be finite, got {v}")
    if positive and v <= 0:
        raise ValueError(f"'{name}' must be positive (> 0), got {v}")
    if not positive and v < 0:
        raise ValueError(f"'{name}' must be non-negative (>= 0), got {v}")


# ---------------------------------------------------------------------------
# 1. Safety stock
# ---------------------------------------------------------------------------

class SafetyStockPolicy(Enum):
    """Available safety-stock calculation methods.

    FIXED_DAYS
        Simplest baseline: safety stock = ceil(daily demand × number of extra
        cover days).  No statistical service-level guarantee.  Good for a
        first prototype or when demand variability is unknown.

    Z_SCORE
        Statistical policy that targets a specific service level.  Requires
        the standard deviation of daily demand and (optionally) the standard
        deviation of lead time.  Formula:
            SS = Z × sqrt(lead_time × σ_demand²  +  avg_demand² × σ_lead²)
        where Z is the service-level z-score (e.g. 1.65 for 95 %).
    """
    FIXED_DAYS = "fixed_days"
    Z_SCORE    = "z_score"


def calculate_safety_stock(
    average_daily_demand: float,
    *,
    policy: SafetyStockPolicy = SafetyStockPolicy.FIXED_DAYS,
    # FIXED_DAYS parameters
    safety_days: float = 1.0,
    # Z_SCORE parameters
    z_score: float = 1.65,
    sigma_demand: float = 0.0,
    lead_time_days: int = 1,
    sigma_lead_time: float = 0.0,
) -> int:
    """Calculate safety stock units using the specified policy.

    Parameters
    ----------
    average_daily_demand : mean forecast demand per day (units/day).
    policy               : SafetyStockPolicy enum member (default: FIXED_DAYS).

    FIXED_DAYS keyword arguments
    ----------------------------
    safety_days : number of extra days of demand to hold as a buffer
                  (default 1.0).  Increase for less reliable suppliers.

    Z_SCORE keyword arguments
    -------------------------
    z_score          : service-level multiplier (default 1.65 ≈ 95 % service).
                       Common values: 1.28 (90 %), 1.65 (95 %), 2.05 (98 %).
    sigma_demand     : standard deviation of daily demand (units/day).
    lead_time_days   : mean lead time in days.
    sigma_lead_time  : standard deviation of lead time in days (0 if constant).

    Returns
    -------
    Safety stock in whole units (always rounded up).

    Examples
    --------
    >>> calculate_safety_stock(20, safety_days=1)
    20
    >>> calculate_safety_stock(20, policy=SafetyStockPolicy.Z_SCORE,
    ...     z_score=1.65, sigma_demand=5, lead_time_days=3)
    15
    """
    _check("average_daily_demand", average_daily_demand)

    if policy == SafetyStockPolicy.FIXED_DAYS:
        _check("safety_days", safety_days, positive=True)
        return math.ceil(average_daily_demand * safety_days)

    elif policy == SafetyStockPolicy.Z_SCORE:
        _check("z_score", z_score, positive=True)
        _check("sigma_demand", sigma_demand)
        _check("lead_time_days", lead_time_days, positive=True)
        _check("sigma_lead_time", sigma_lead_time)
        variance = (lead_time_days * sigma_demand ** 2
                    + average_daily_demand ** 2 * sigma_lead_time ** 2)
        return math.ceil(z_score * math.sqrt(variance))

    else:
        raise ValueError(f"Unknown safety-stock policy: {policy!r}")


# ---------------------------------------------------------------------------
# 2. Expected demand during lead time
# ---------------------------------------------------------------------------

def expected_lead_time_demand(
    forecast_daily_demand: "float | Sequence[float]",
    lead_time_days: int,
) -> float:
    """Return the total expected demand over the supplier lead-time horizon.

    Two modes
    ---------
    Scalar mode
        Pass a single float representing the *average* daily forecast.
        The function multiplies it by ``lead_time_days``.
        Example: demand=20 units/day, lead_time=3 days → 60 units.

    Sequence mode
        Pass a list/array of per-day forecasts.  The function sums the
        first ``lead_time_days`` values and does NOT multiply again.
        Example: [10, 12, 14] with lead_time=3 → 36 units.
        The list must contain at least ``lead_time_days`` entries.

    Parameters
    ----------
    forecast_daily_demand : average daily demand (float) OR list of daily
                            forecasts (Sequence[float]).
    lead_time_days        : supplier lead time in days (must be >= 1).

    Returns
    -------
    Total expected demand in units over the lead-time window (float).

    Raises
    ------
    ValueError : if lead_time_days < 1, any forecast value is invalid, or the
                 sequence is shorter than lead_time_days.
    """
    _check("lead_time_days", lead_time_days, positive=True)

    if isinstance(forecast_daily_demand, (int, float)):
        _check("forecast_daily_demand", forecast_daily_demand)
        return float(forecast_daily_demand) * int(lead_time_days)

    # --- sequence mode ---
    values = list(forecast_daily_demand)
    if len(values) < lead_time_days:
        raise ValueError(
            f"Forecast sequence has {len(values)} entries but lead_time_days "
            f"is {lead_time_days}; the list must cover the full lead-time horizon."
        )
    horizon = values[:lead_time_days]
    for i, v in enumerate(horizon):
        _check(f"forecast value at index {i}", v)
    return float(sum(horizon))


# ---------------------------------------------------------------------------
# 3. Reorder point
# ---------------------------------------------------------------------------

def calculate_reorder_point(
    lead_time_demand: float,
    safety_stock_units: float,
) -> int:
    """Calculate the Reorder Point (ROP).

    ROP = Expected Demand During Lead Time + Safety Stock

    When inventory position drops to or below ROP, a new order should be
    placed so that stock arrives before running out.

    Parameters
    ----------
    lead_time_demand   : total expected demand over the lead-time window.
    safety_stock_units : buffer stock to absorb variability.

    Returns
    -------
    ROP in whole units (rounded up).

    Example
    -------
    Lead-time demand = 60, safety stock = 15 → ROP = 75.
    """
    _check("lead_time_demand", lead_time_demand)
    _check("safety_stock_units", safety_stock_units)
    return math.ceil(lead_time_demand + safety_stock_units)


# ---------------------------------------------------------------------------
# 4. Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------

def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    annual_holding_cost_per_unit: float,
) -> int:
    """Calculate Economic Order Quantity (EOQ).

    EOQ = sqrt( (2 × Annual Demand × Ordering Cost) / Annual Holding Cost )

    EOQ minimises the total of ordering cost and holding (carrying) cost.
    It assumes constant demand, known costs, and instantaneous replenishment —
    a simplified model that is a useful starting point.

    Parameters
    ----------
    annual_demand                : total units expected to be sold per year.
                                   Must be >= 0.  Returns 0 if demand is 0.
    ordering_cost                : fixed cost per purchase order (₹/order).
                                   Must be > 0.
    annual_holding_cost_per_unit : cost to hold one unit for one year (₹/unit/year).
                                   Must be > 0.  Usually 15–30 % of unit cost.

    Returns
    -------
    EOQ in whole units (rounded up).

    Example
    -------
    Annual demand 7 300, ordering cost ₹100, holding cost ₹20/unit/year
    → EOQ = sqrt(2 × 7300 × 100 / 20) = sqrt(73 000) ≈ 270.19 → 271 units.
    """
    _check("annual_demand", annual_demand)          # >= 0 allowed
    _check("ordering_cost", ordering_cost, positive=True)
    _check("annual_holding_cost_per_unit", annual_holding_cost_per_unit, positive=True)
    if annual_demand == 0:
        return 0
    return math.ceil(math.sqrt(2 * annual_demand * ordering_cost / annual_holding_cost_per_unit))


# ---------------------------------------------------------------------------
# 5. Inventory position
# ---------------------------------------------------------------------------

def calculate_inventory_position(
    on_hand: float,
    on_order: float = 0.0,
    backorders: float = 0.0,
) -> float:
    """Calculate the current inventory position.

    Inventory Position = On-Hand + On-Order − Backorders

    Using *inventory position* instead of just on-hand stock prevents
    double-ordering when a replenishment is already in transit.

    Parameters
    ----------
    on_hand    : units physically in the warehouse right now.
    on_order   : units already ordered from the supplier but not yet received.
                 If this information is not available, leave it at 0 and note
                 the assumption in the recommendation reason.
    backorders : units owed to customers that have not been fulfilled yet.
                 If not tracked, leave at 0.

    Returns
    -------
    Inventory position (float).  Can be lower than on_hand when backorders exist.

    Example
    -------
    on_hand=100, on_order=20, backorders=5 → position = 115.
    """
    for name, val in [("on_hand", on_hand), ("on_order", on_order), ("backorders", backorders)]:
        _check(name, val)
    return float(on_hand) + float(on_order) - float(backorders)


# ---------------------------------------------------------------------------
# 6. Recommended order quantity
# ---------------------------------------------------------------------------

def recommend_order_quantity(
    inventory_position: float,
    reorder_point: float,
    eoq_units: int,
    minimum_order_quantity: int,
    storage_capacity_units: "int | None" = None,
    on_hand_units: "float | None" = None,
) -> dict:
    """Decide whether to order and how many units to recommend.

    Decision rules (in order)
    -------------------------
    1. If inventory_position > reorder_point → no order needed.
    2. If an order is needed, start with max(EOQ, MOQ) as the desired quantity.
    3. If storage_capacity_units is provided:
       a. Calculate remaining room = capacity − on_hand_units.
       b. If room < MOQ → order is *blocked* (physically impossible to receive
          a minimum-sized delivery); flag for manager action.
       c. Otherwise cap the order at available room.
    4. Return a structured result dict with a plain-English reason.

    ⚠ HUMAN APPROVAL REQUIRED before placing a real purchase order.
    The recommended quantity is a suggestion; a manager must confirm it.

    Parameters
    ----------
    inventory_position     : current inventory position (on-hand + on-order − backorders).
    reorder_point          : trigger level calculated by calculate_reorder_point().
    eoq_units              : optimal order size from calculate_eoq().
    minimum_order_quantity : smallest order the supplier will accept.
    storage_capacity_units : maximum units the store can hold (optional).
    on_hand_units          : units currently in the warehouse (required if
                             storage_capacity_units is provided).

    Returns
    -------
    dict with keys:
        replenishment_required   (bool)
        recommended_order_quantity (int)  — 0 means no order or blocked
        recommendation_reason    (str)   — plain-English explanation
        blocked_by_capacity      (bool)
    """
    _check("inventory_position", inventory_position)
    _check("reorder_point", reorder_point)
    _check("eoq_units", eoq_units)
    _check("minimum_order_quantity", minimum_order_quantity, positive=True)

    # --- Rule 1: no order needed ---
    if inventory_position > reorder_point:
        return {
            "replenishment_required": False,
            "recommended_order_quantity": 0,
            "recommendation_reason": (
                f"No order needed. Inventory position ({inventory_position:.0f}) "
                f"is above the reorder point ({reorder_point:.0f})."
            ),
            "blocked_by_capacity": False,
        }

    # --- Rule 2: desired quantity ---
    desired = max(int(eoq_units), int(minimum_order_quantity))

    # --- Rule 3: storage capacity check ---
    if storage_capacity_units is not None:
        _check("storage_capacity_units", storage_capacity_units, positive=True)
        if on_hand_units is None:
            raise ValueError(
                "'on_hand_units' must be provided when 'storage_capacity_units' is given."
            )
        _check("on_hand_units", on_hand_units)
        room = max(0, int(storage_capacity_units) - int(on_hand_units))

        if room < int(minimum_order_quantity):
            return {
                "replenishment_required": True,
                "recommended_order_quantity": 0,
                "recommendation_reason": (
                    f"Order BLOCKED. Remaining storage capacity ({room} units) is less "
                    f"than the supplier's minimum order quantity ({minimum_order_quantity} units). "
                    f"Manager action required: either clear storage space or negotiate with supplier."
                ),
                "blocked_by_capacity": True,
            }

        qty = min(desired, room)
    else:
        qty = desired

    return {
        "replenishment_required": True,
        "recommended_order_quantity": int(qty),
        "recommendation_reason": (
            f"Order recommended: {qty} units (EOQ={eoq_units}, MOQ={minimum_order_quantity}). "
            f"Inventory position ({inventory_position:.0f}) is at or below the reorder point "
            f"({reorder_point:.0f}). Human approval required before placing this order."
        ),
        "blocked_by_capacity": False,
    }
