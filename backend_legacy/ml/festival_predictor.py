from typing import Optional

# ==========================================
# FESTIVAL PREDICTOR
# Deterministic module — no ML needed here.
# Uses risk score + festival context to compute
# actionable stockout forecasts for the retailer.
# ==========================================


def compute_stockout_forecast(
    current_stock: int,
    daily_sales: int,
    festival_ctx: dict,
    lead_time_days: int = 3,
) -> dict:
    """
    Calculates how many days until the shop runs out of stock,
    how much to order, and how urgently.

    Args:
        current_stock:  units currently on shelf
        daily_sales:    average units sold per day (normal conditions)
        festival_ctx:   output from festival_calendar.get_festival_context()
        lead_time_days: days between placing an order and receiving it

    Returns:
        {
            days_until_stockout:     float
            effective_daily_demand:  float  (sales × spike factor)
            recommended_order_qty:   int
            restock_urgency:         str    (IMMEDIATE | URGENT | SOON | ROUTINE)
            festival_name:           str | None
            days_to_festival:        int
            demand_multiplier:       float
            restock_deadline:        str    (human-readable)
        }
    """
    spike_factor      = festival_ctx["spike_factor"]       # 1.0 if not affected
    days_to_festival  = festival_ctx["days_to_festival"]
    in_prep_window    = festival_ctx["in_prep_window"]
    festival_name     = festival_ctx["festival_name"]

    # Effective demand: normal sales × festival spike (if in prep window)
    effective_daily_demand = daily_sales * (spike_factor if in_prep_window else 1.0)
    effective_daily_demand = max(effective_daily_demand, 0.1)  # avoid division by zero

    # Days until shelves go empty at current demand rate
    days_until_stockout = current_stock / effective_daily_demand

    # How much stock is needed to cover the festival window + 7 day buffer
    festival_window = min(days_to_festival + 7, 30)
    target_stock = int(effective_daily_demand * festival_window * 1.2)  # 20% safety buffer
    recommended_order_qty = max(0, target_stock - current_stock)

    # Urgency classification
    if days_until_stockout <= lead_time_days:
        urgency = "IMMEDIATE"   # Will run out before order can even arrive
    elif days_until_stockout < days_to_festival and in_prep_window:
        urgency = "URGENT"      # Will run out before the festival
    elif days_until_stockout < days_to_festival + 7:
        urgency = "SOON"        # Cutting it close
    else:
        urgency = "ROUTINE"     # Comfortable window

    # Human-readable restock deadline
    if urgency == "IMMEDIATE":
        restock_deadline = "TODAY"
    elif urgency == "URGENT":
        restock_deadline = f"Within {int(days_until_stockout - lead_time_days)} days"
    elif festival_name:
        restock_deadline = f"Before {festival_name} ({days_to_festival} days away)"
    else:
        restock_deadline = "No rush"

    return {
        "days_until_stockout":    round(days_until_stockout, 1),
        "effective_daily_demand": round(effective_daily_demand, 1),
        "recommended_order_qty":  recommended_order_qty,
        "restock_urgency":        urgency,
        "festival_name":          festival_name,
        "days_to_festival":       days_to_festival,
        "demand_multiplier":      round(spike_factor, 1),
        "restock_deadline":       restock_deadline,
    }
