from datetime import date
from typing import Optional

# ==========================================
# FESTIVAL CALENDAR
# Maps Indian festivals (relevant to Telangana/Jangaon)
# to their affected products and historical demand spike factors.
#
# spike_factor: how much daily demand multiplies during this festival
# prep_days:    how many days before the festival demand starts rising
# ==========================================

FESTIVAL_CALENDAR = {
    "Sankranti": {
        "month": 1, "day": 14,
        "products": ["Rice (50kg)", "Pulses Mix (25kg)"],
        "spike_factor": 1.9,
        "prep_days": 14,
    },
    "Ugadi": {
        "month": 3, "day": 22,
        "products": ["Rice (50kg)", "Sugar (50kg)"],
        "spike_factor": 2.1,
        "prep_days": 10,
    },
    "Eid": {
        "month": 4, "day": 10,
        "products": ["Wheat Flour (40kg)", "Sugar (50kg)", "Cooking Oil (15L)"],
        "spike_factor": 2.0,
        "prep_days": 14,
    },
    "Dussehra": {
        "month": 10, "day": 2,
        "products": ["Wheat Flour (40kg)", "Sugar (50kg)"],
        "spike_factor": 1.7,
        "prep_days": 7,
    },
    "Diwali": {
        "month": 10, "day": 28,
        "products": ["Cooking Oil (15L)", "Sugar (50kg)", "Wheat Flour (40kg)"],
        "spike_factor": 2.8,
        "prep_days": 21,
    },
}


def get_festival_context(
    product_name: str,
    target_date: Optional[date] = None
) -> dict:
    """
    Given a product name and date, finds the nearest upcoming festival
    and returns a context dict the model uses as features.

    Args:
        product_name: e.g. "Rice (50kg)", "Cooking Oil (15L)"
        target_date:  defaults to today

    Returns:
        {
            festival_name:      str | None
            days_to_festival:   int   (90 if no festival nearby)
            spike_factor:       float (1.0 if product not affected)
            product_affinity:   float (1.0 if affected, 0.0 if not)
            affected_products:  list[str]
            in_prep_window:     bool  (True if within prep_days of festival)
        }
    """
    if target_date is None:
        target_date = date.today()

    nearest = None
    min_days = float("inf")

    for name, info in FESTIVAL_CALENDAR.items():
        for year in [target_date.year, target_date.year + 1]:
            try:
                f_date = date(year, info["month"], info["day"])
                days_away = (f_date - target_date).days
                if 0 <= days_away < min_days:
                    min_days = days_away
                    nearest = {
                        "festival_name":    name,
                        "days_to_festival": days_away,
                        "spike_factor":     info["spike_factor"],
                        "product_affinity": 1.0 if product_name in info["products"] else 0.0,
                        "affected_products": info["products"],
                        "in_prep_window":   days_away <= info["prep_days"],
                    }
            except ValueError:
                continue

    if nearest is None:
        return {
            "festival_name":    None,
            "days_to_festival": 90,
            "spike_factor":     1.0,
            "product_affinity": 0.0,
            "affected_products": [],
            "in_prep_window":   False,
        }

    # If product is not affected, spike factor is 1.0 (no spike)
    if nearest["product_affinity"] == 0.0:
        nearest["spike_factor"] = 1.0

    return nearest
