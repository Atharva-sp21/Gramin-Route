from pydantic import BaseModel
from typing import Optional

# ==========================================
# PYDANTIC SCHEMAS
# Extracted from main.py for separation of concerns.
# ==========================================


class RetailerInput(BaseModel):
    shop_id:        str
    lat:            float
    lon:            float
    current_stock:  int
    daily_sales:    int   = 5
    lead_time_days: int   = 3
    profit_margin:  float = 20.0
    shelf_life:     int   = 30
    credit_score:   int   = 700
    # Product name drives festival calendar lookup
    # Default matches the most common kirana staple
    product_name:   str   = "Rice (50kg)"


class PendingOrder(BaseModel):
    shop_id:      str
    lat:          float
    lon:          float
    qty_needed:   int
    # Handle both naming conventions from the frontend
    retailer_id:  Optional[str]   = None
    retailer_lat: Optional[float] = None
    retailer_lon: Optional[float] = None
