# Smart Store Operational Layer Documentation

**Author:** Punith  
**Project:** Smart Store: An AI-Powered Inventory and Storage Intelligence Platform  
**Modules:** `src/synthetic_operational_data.py`, `src/replenishment.py`

---

## 1. Purpose of the Operational Layer

The raw canonical dataset (`dataset/retail_store_inventory.csv`) contains sales records, inventory snapshots, prices, and promotions. However, in real-world retail supply chains, automated replenishment decisions cannot be made from sales and inventory levels alone. 

To determine **when to reorder**, **how much to order**, and **whether an order is physically and commercially feasible**, the system needs operational parameters such as:
- How long a supplier takes to deliver items (lead time)
- The smallest delivery volume a supplier allows (minimum order quantity / MOQ)
- The costs of placing orders and holding excess stock (EOQ parameters)
- Storage facility limits preventing warehouse overflows

The **operational layer** bridges the gap between raw data/machine learning demand forecasts and actionable business recommendations.

---

## 2. Synthetic Operational Fields

Because the public Kaggle dataset does not include supplier contracts or warehouse specifications, we generate realistic, synthetic operational attributes using `src/synthetic_operational_data.py`.

A fixed random seed (`--seed 42`) guarantees **100% reproducibility**. Every run produces identical supplier parameters for each product ID across all stores and dates.

### Column Definitions

| Column Name | Type | Unit | Description & Realistic Assumptions |
|---|---|---|---|
| `supplier_id` | String | Identifier | Unique supplier code (e.g., `SUP001` to `SUP008`). Assigned consistently per product. |
| `lead_time_days` | Integer | Days | The delivery duration from order placement to goods receipt. Uniform random: 2 to 10 days. |
| `minimum_order_quantity` (MOQ) | Integer | Units | Smallest allowable batch size required by the vendor contract. Uniform random: 10 to 100 units. |
| `unit_cost` | Float | Currency (₹) | Procurement cost per unit paid to the supplier. Uniform random: ₹5.00 to ₹500.00. |
| `ordering_cost` | Float | Currency (₹/order) | Fixed administrative and logistics cost to process one purchase order. Uniform random: ₹50.00 to ₹250.00. |
| `annual_holding_cost_per_unit` | Float | Currency (₹/unit/yr) | Cost of holding 1 unit of stock for a full year (capital cost, shrinkage, insurance, warehousing). Modeled as 15% to 30% of `unit_cost`. |
| `storage_capacity_units` | Integer | Units | Maximum physical storage capacity allocated for this product category at the store warehouse. Uniform random: 100 to 2,000 units. |
| `operational_data_is_synthetic` | Boolean | Flag | Explicit boolean flag (`True`) marking that these fields are synthetic simulation assumptions, not verified supplier contracts. |

---

## 3. How the Replenishment Engine Makes Decisions

The replenishment engine (`src/replenishment.py`) executes the following end-to-end pipeline:

```
[Latest Inventory Snapshot] + [Forecast Pipeline] + [Operational Parameters]
                                     │
                                     ▼
                      1. Calculate Lead-Time Demand (LTD)
                      2. Calculate Safety Stock (SS)
                      3. Determine Reorder Point (ROP = LTD + SS)
                      4. Determine Inventory Position (IP = On-Hand + On-Order - Backorders)
                      5. Compare IP vs ROP
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
     IP > ROP                                             IP <= ROP
  [No Order Needed]                                 [Order Triggered]
  - Quantity = 0                                    - Calculate EOQ
  - Risk: LOW                                       - Start with max(EOQ, MOQ)
                                                    - Check Storage Capacity:
                                                        * room < MOQ  -> BLOCKED (Risk: HIGH)
                                                        * room >= MOQ -> Cap at room (qty = min(desired, room))
                                                    - Calculate Estimated Cost
                                                    - Flag for Human Approval
```

### Risk Level Definitions
- **`HIGH`**: Assigned when:
  1. An order is blocked due to insufficient warehouse capacity, OR
  2. Inventory position is already strictly below expected lead-time demand (`inventory_position < expected_lead_time_demand`), meaning the store is at imminent risk of running out of stock before a delivery could arrive.
- **`MODERATE`**: Replenishment is required (`inventory_position <= reorder_point`), but current stock is still sufficient to cover expected demand during the lead-time window.
- **`LOW`**: Inventory position is safely above the reorder point. No order is needed.

---

## 4. Integration with the Forecasting Team

The replenishment engine supports two flexible forecast formats from the modeling team:

### Option A: Scalar Daily Forecasts (Default)
A CSV containing:
- `Product ID`
- `Store ID`
- `forecast_units`: The average predicted daily demand in units for that product-store pair.

**Calculation:**
$$\text{Expected Lead-Time Demand} = \text{forecast\_units} \times \text{lead\_time\_days}$$

### Option B: Time-Series Daily Forecasts (Multi-Day Horizon)
For advanced models predicting individual future dates:
- `Product ID`
- `Store ID`
- `forecast_date`: Date string (`YYYY-MM-DD`).
- `forecast_units`: Daily predicted demand for that specific date.

**Calculation:**
The engine sorts dates chronologically and **sums** the first $N$ days where $N = \text{lead\_time\_days}$:
$$\text{Expected Lead-Time Demand} = \sum_{t=1}^{\text{lead\_time\_days}} \text{forecast\_units}_t$$
*Note:* The engine never multiplies multi-day summed forecasts by lead time again.

---

## 5. Assumptions, Limitations, and Human Approval

1. **Target Leakage Warning Regarding `Demand Forecast`:**  
   The raw dataset contains a column named `Demand Forecast` which has near-perfect correlation (0.99+) with `Units Sold`. As established in our team's EDA findings, this feature exhibits severe target leakage and must **never** be used as future demand without formal validation. The replenishment engine only accepts forecasts from our team's leakage-safe modeling pipeline or synthetic demo fixtures.

2. **Annual Demand Approximation:**  
   In the absence of a separate multi-year sales model, annual demand is estimated as $\text{daily forecast} \times 365$. If seasonal annual projections become available, they can be plugged directly into the EOQ formula.

3. **Incoming Orders and Backorders:**  
   The canonical dataset only provides historical `Inventory Level`. It does not track open purchase orders (`on_order`) or customer backorders. In this version, `inventory_position` defaults to physical on-hand stock and explicitly records this assumption in the `inventory_position_note` field.

4. **Why Human Approval is Mandatory:**  
   Algorithms optimize based on numbers, but real retail environments encounter real-world exceptions:
   - Cash flow constraints and payment terms
   - Unannounced supplier stockouts or price revisions
   - Upcoming store renovations or shelf-space changes
   - Perishable shelf-life constraints
   
   Therefore, **every recommendation is an advisory suggestion requiring manager sign-off** before generating an actual vendor purchase order.
