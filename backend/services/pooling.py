import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
from typing import List, Dict

# ==========================================
# POOLING ENGINE
# Extracted from main.py into its own service.
#
# Uses DBSCAN (Haversine metric) to cluster nearby
# orders into delivery pools — replacing the manual
# greedy Haversine loop from v1.
#
# eps = 3km converted to radians for sklearn.
# Shops labelled -1 (noise/isolated) get solo delivery.
# ==========================================


def generate_pools(orders: List[Dict]) -> List[Dict]:
    """
    Groups a list of normalised order dicts into delivery pools.

    Each order dict must have: id, qty, lat, lon.

    Returns a list of pool dicts with:
        pool_id, shops, total_qty, center_lat, center_lon,
        radius_km, discount, is_solo
    """
    if not orders:
        return []

    coords = np.radians([[o["lat"], o["lon"]] for o in orders])

    db = DBSCAN(
        eps=3.0 / 6371.0,       # 3km in radians
        min_samples=1,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords)

    labels = db.labels_
    pools  = []

    for label in set(labels):
        indices   = [i for i, l in enumerate(labels) if l == label]
        pool_shops = [orders[i] for i in indices]

        total_qty  = sum(s["qty"] for s in pool_shops)
        center_lat = float(np.mean([s["lat"] for s in pool_shops]))
        center_lon = float(np.mean([s["lon"] for s in pool_shops]))

        # Max radius within the pool (Haversine)
        max_radius = 0.0
        R = 6371.0
        for s in pool_shops:
            dlat = radians(s["lat"] - center_lat)
            dlon = radians(s["lon"] - center_lon)
            a = (sin(dlat / 2) ** 2
                 + cos(radians(center_lat)) * cos(radians(s["lat"])) * sin(dlon / 2) ** 2)
            max_radius = max(max_radius, R * 2 * atan2(sqrt(a), sqrt(1 - a)))

        pools.append({
            "pool_id":    f"POOL-{label + 1:03d}" if label >= 0 else f"SOLO-{indices[0]:03d}",
            "shops":      [s["id"] for s in pool_shops],
            "total_qty":  total_qty,
            "center_lat": round(center_lat, 6),
            "center_lon": round(center_lon, 6),
            "radius_km":  round(max_radius, 3),
            "discount":   "15% WHOLESALE" if total_qty > 50 else "STANDARD",
            "is_solo":    label == -1,
        })

    return pools


def normalise_orders(raw_orders) -> List[Dict]:
    """
    Normalises PendingOrder Pydantic objects (or dicts) to the
    flat dict shape generate_pools() expects.
    Handles both naming conventions from the frontend.
    """
    clean = []
    for o in raw_orders:
        # Support both Pydantic objects and dicts
        if hasattr(o, "__dict__"):
            o = o.__dict__

        clean.append({
            "id":  o.get("shop_id") or o.get("retailer_id"),
            "qty": o.get("qty_needed", 0),
            "lat": o.get("retailer_lat") or o.get("lat", 0.0),
            "lon": o.get("retailer_lon") or o.get("lon", 0.0),
        })
    return clean
