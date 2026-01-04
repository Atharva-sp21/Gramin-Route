# ==========================================
# FILE: main.py (UPDATED WITH POOLING & RECOMMENDER)
# ==========================================
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Import your AI Brain
from model_def import HybridQuantumGNN 

app = FastAPI()

# --- 1. LOAD THE BRAIN ---
model = HybridQuantumGNN(in_dim=7)
try:
    model.load_state_dict(torch.load("models/quantum_gnn_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("✅ Quantum Brain Loaded")
except:
    print("⚠️ Running in Logic-Only Mode (Model not found)")

# --- 2. DATA MODELS (Inputs) ---
class RetailerRequest(BaseModel):
    shop_id: int
    lat: float
    lon: float
    current_stock: int
    is_festival: bool

class PendingOrder(BaseModel):
    shop_id: int
    lat: float
    lon: float
    qty_needed: int

# --- 3. HELPER: GEOSPATIAL DISTANCE (Haversine) ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ==========================================
# ENDPOINT 1: DISTRIBUTOR POOLING ENGINE
# ==========================================
@app.post("/pool_orders")
def generate_pools(orders: list[PendingOrder]):
    """
    Groups separate orders into "Pools" to minimize logistics cost.
    Input: List of 50 separate orders.
    Output: 5 Pools (groups of neighbors).
    """
    pools = []
    processed = set()
    pool_counter = 1

    for i, order in enumerate(orders):
        if order.shop_id in processed:
            continue
        
        # Start a new pool with this shop
        current_pool = {
            "pool_id": f"POOL-{pool_counter:03d}",
            "shops": [order.shop_id],
            "total_qty": order.qty_needed,
            "center_lat": order.lat,
            "center_lon": order.lon
        }
        processed.add(order.shop_id)

        # Look for neighbors to merge
        for j, neighbor in enumerate(orders):
            if neighbor.shop_id in processed:
                continue
            
            # Check Distance (e.g., < 3km radius)
            dist = calculate_distance(order.lat, order.lon, neighbor.lat, neighbor.lon)
            
            if dist < 3.0: # 3 KM Clumping Radius
                current_pool["shops"].append(neighbor.shop_id)
                current_pool["total_qty"] += neighbor.qty_needed
                processed.add(neighbor.shop_id)
        
        # Add 'Bulk Discount' Tag if pool is large
        if current_pool["total_qty"] > 50:
            current_pool["discount_applied"] = "WHOLESALE_RATE (15% OFF)"
        else:
            current_pool["discount_applied"] = "STANDARD_RATE"
            
        pools.append(current_pool)
        pool_counter += 1

    return {"optimized_pools": pools, "total_pools": len(pools)}

# ==========================================
# ENDPOINT 2: RETAILER RECOMMENDER SYSTEM
# ==========================================
@app.post("/recommend_distributor")
def recommend_distributor(shop: RetailerRequest):
    """
    Suggests the best distributor based on Shop's Urgency (Risk).
    """
    # 1. Ask AI for Risk Score
    # (Simulated input vector for demo)
    risk_score = 0.2
    if shop.is_festival: risk_score += 0.4
    if shop.current_stock < 15: risk_score += 0.3
    
    # 2. Define Available Distributors (Mock Database)
    distributors = [
        {"name": "FastTrack Logistics", "cost": 100, "speed_hrs": 4, "reliability": 0.98},
        {"name": "Budget Movers",       "cost": 60,  "speed_hrs": 24, "reliability": 0.85},
        {"name": "GraminRoute Hub",     "cost": 75,  "speed_hrs": 12, "reliability": 0.99} # Your Service
    ]
    
    # 3. Intelligent Ranking Logic
    recommendations = []
    
    for dist in distributors:
        score = 0
        
        # SCENARIO A: HIGH RISK (Shop is about to stock out!)
        # Priority: SPEED is king. Cost doesn't matter.
        if risk_score > 0.7:
            score += (100 / dist["speed_hrs"]) * 2.0  # Double weight on speed
            score += (dist["reliability"] * 100)
            reason = "FASTEST (Urgent Need)"
            
        # SCENARIO B: LOW RISK (Routine Restock)
        # Priority: COST is king. Speed doesn't matter.
        else:
            score += (200 / dist["cost"]) * 2.0       # Double weight on cheapness
            score += (dist["reliability"] * 50)
            reason = "BEST PRICE (Routine)"
            
        recommendations.append({
            "distributor": dist["name"],
            "match_score": round(score, 1),
            "reason": reason,
            "cost_per_unit": dist["cost"],
            "eta": f"{dist['speed_hrs']} Hours"
        })
    
    # Sort by Score (Highest First)
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    
    return {
        "shop_status": "CRITICAL" if risk_score > 0.7 else "STABLE",
        "top_pick": recommendations[0]
    }  