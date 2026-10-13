# Smart Store — Inventory Formula Reference Guide

**Author:** Punith  
**Module:** `src/inventory_formulas.py`  
**Tests:** `tests/test_inventory_formulas.py`

This guide explains the mathematical formulas, behavioral logic, edge-case handling, and numerical examples used in the Smart Store replenishment calculations.

---

## 1. Safety Stock ($SS$)

### Purpose
Safety stock acts as a protective buffer to prevent stockouts caused by demand spikes or supplier delivery delays.

### Policy 1: Fixed Days Coverage (`FIXED_DAYS` — Default Policy)
$$SS = \lceil \text{Average Daily Demand} \times \text{Safety Days} \rceil$$
- **When to use:** Early prototyping, stable demand patterns, or when demand variance ($\sigma$) is not yet modeled.
- **Example:**
  - Average daily demand = 20 units/day
  - Safety buffer = 1.5 days
  - $SS = \lceil 20 \times 1.5 \rceil = \lceil 30.0 \rceil = 30\text{ units}$

### Policy 2: Statistical Z-Score Coverage (`Z_SCORE` — Service-Level Policy)
$$SS = \left\lceil Z \times \sqrt{L \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_L^2} \right\rceil$$
Where:
- $Z$ = Normal distribution quantile for target cycle service level (e.g., $1.28 \approx 90\%$, $1.65 \approx 95\%$, $2.05 \approx 98\%$)
- $L$ = Average supplier lead time (days)
- $\sigma_d$ = Standard deviation of daily forecast demand
- $\bar{d}$ = Mean daily forecast demand
- $\sigma_L$ = Standard deviation of lead time (0 if lead time is constant)

---

## 2. Expected Demand During Lead Time ($LTD$)

### Purpose
Represents the total volume of inventory that customers will consume during the days between placing a purchase order and receiving delivery.

### Calculation Modes
1. **Scalar Mode (Single Average Forecast):**
   $$LTD = \text{forecast\_daily\_demand} \times \text{lead\_time\_days}$$
   - *Example:* Average forecast = $25\text{ units/day}$, Lead time = $4\text{ days} \implies LTD = 25 \times 4 = 100\text{ units}$.

2. **Time-Series Mode (Multi-Date Forecast Horizon):**
   $$LTD = \sum_{t=1}^{\text{lead\_time\_days}} \text{daily\_forecast}_t$$
   - *Example:* Lead time = $3\text{ days}$, Daily predictions = $[20, 25, 30] \implies LTD = 20 + 25 + 30 = 75\text{ units}$.
   - *Important Rule:* The sum is **never** multiplied by lead time again.

---

## 3. Reorder Point ($ROP$)

### Purpose
The threshold inventory position that triggers a purchase order.
$$\text{ROP} = \lceil LTD + SS \rceil$$

### Logic
If current inventory position drops to or below $ROP$, an order is placed immediately so that replacement stock arrives right as the safety buffer is reached.

### Worked Example:
- Forecast daily demand: $15\text{ units/day}$
- Supplier lead time: $4\text{ days}$ ($LTD = 15 \times 4 = 60\text{ units}$)
- Safety stock: $15\text{ units}$ ($1\text{ day of buffer}$)
- $\text{ROP} = 60 + 15 = 75\text{ units}$.

---

## 4. Economic Order Quantity ($EOQ$)

### Purpose
Determines the cost-optimal batch size to order by balancing fixed ordering costs against inventory holding (carrying) costs.

$$\text{EOQ} = \left\lceil \sqrt{\frac{2 \times D \times S}{H}} \right\rceil$$

Where:
- $D$ = Annual demand (units/year). Modeled as $\text{daily forecast} \times 365$.
- $S$ = Fixed cost per purchase order ($\text{ordering\_cost}$, ₹/order).
- $H$ = Annual holding cost per unit ($\text{annual\_holding\_cost\_per\_unit}$, ₹/unit/year).

### Properties & Edge Cases:
- If annual demand $D = 0 \implies EOQ = 0$.
- Both $S$ and $H$ must be strictly positive ($> 0$).
- Output is always rounded up to an integer unit ($\lceil \cdot \rceil$).

### Worked Example:
- Daily demand = $20\text{ units/day} \implies D = 20 \times 365 = 7,300\text{ units/year}$
- Fixed order cost $S = ₹100.00$
- Unit holding cost $H = ₹20.00/\text{unit/year}$
- $\text{EOQ} = \left\lceil \sqrt{\frac{2 \times 7300 \times 100}{20}} \right\rceil = \lceil \sqrt{73,000} \rceil = \lceil 270.185 \rceil = 271\text{ units}$.

---

## 5. Inventory Position ($IP$)

### Purpose
Provides the true operational view of available stock, preventing costly double-orders while orders are in transit.

$$\text{Inventory Position} = \text{On-Hand} + \text{On-Order} - \text{Backorders}$$

- **On-Hand:** Units physically present in the store/warehouse.
- **On-Order:** Units already purchased and currently in transit from the supplier.
- **Backorders:** Committed units owed to customers that cannot yet be fulfilled.

*Dataset Note:* In datasets where on-order and backorders are unrecorded, $IP = \text{On-Hand}$, and an explanatory note is recorded.

---

## 6. Recommended Order Quantity Decision Matrix

The engine combines $IP$, $ROP$, $EOQ$, $MOQ$, and Storage Capacity using this exact decision hierarchy:

```
Step 1: Check Reorder Trigger
  Is Inventory Position > ROP?
  ├── YES ──> Replenishment Required = False, Recommended Qty = 0
  └── NO  ──> Replenishment Required = True (Proceed to Step 2)

Step 2: Balance Economic Batch vs Vendor MOQ
  Desired Qty = max(EOQ, MOQ)

Step 3: Evaluate Physical Storage Capacity
  Remaining Room = max(0, Storage Capacity - On-Hand)
  Is Remaining Room < MOQ?
  ├── YES ──> Order is BLOCKED by capacity!
  │           Recommended Qty = 0
  │           Flag: blocked_by_capacity = True
  │           Risk: HIGH (Immediate manager intervention needed)
  └── NO  ──> Feasible order:
              Recommended Qty = min(Desired Qty, Remaining Room)
              Flag: blocked_by_capacity = False

Step 4: Human-in-the-Loop Governance
  Every valid order is flagged: "Human approval required before placing this order."
```

---

## 7. Comprehensive Summary of Examples

| Scenario | On-Hand | LTD | SS | ROP | EOQ | MOQ | Capacity | Outcome | Reason |
|---|---|---|---|---|---|---|---|---|---|
| **Healthy Stock** | 200 | 60 | 20 | 80 | 150 | 50 | 1,000 | Order 0 | $IP (200) > ROP (80)$. Stock is sufficient. |
| **Normal Reorder** | 50 | 60 | 20 | 80 | 150 | 50 | 1,000 | Order 150 | $IP (50) \le ROP (80)$. $EOQ (150) \ge MOQ (50)$. Fits in storage. |
| **MOQ Override** | 70 | 60 | 20 | 80 | 30 | 100 | 1,000 | Order 100 | $IP \le ROP$. $EOQ (30) < MOQ (100)$, raised to vendor minimum 100. |
| **Capacity Capped** | 50 | 60 | 20 | 80 | 300 | 50 | 200 | Order 150 | Room is $200 - 50 = 150$. Desired 300 capped to 150. |
| **Capacity Blocked** | 790 | 600 | 200 | 800 | 200 | 50 | 800 | Order 0 (BLOCKED) | Room is $800 - 790 = 10 < MOQ (50)$. Cannot legally order. |
