from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import List

# Import your AI Brain (Optional wrapper for safety)
try:
    from model_def import HybridQuantumGNN 
except ImportError:
    HybridQuantumGNN = None

app = FastAPI()

# --- 0. CORS CONFIGURATION (CRITICAL FOR REACT) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev (change to localhost:3000 in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD THE BRAIN ---
model = None
if HybridQuantumGNN:
    model = HybridQuantumGNN(in_dim=7)
    try:
        model.load_state_dict(torch.load("model/quantum_gnn_model.pth", map_location=torch.device('cpu')))
        model.eval()
        print("✅ Quantum Brain Loaded")
    except:
        print("⚠️ Running in Logic-Only Mode (Model file not found)")
else:
    print("⚠️ Running in Logic-Only Mode (model_def.py not found)")

# --- 2. DATA MODELS (Inputs) ---
class RetailerRequest(BaseModel):
    shop_id: str   # <--- CHANGED FROM int TO str
    lat: float
    lon: float
    current_stock: int
    is_festival: bool = False

class PendingOrder(BaseModel):
    shop_id: str   # <--- CHANGED FROM int TO str
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
def generate_pools(orders: List[PendingOrder]):
    """
    Groups separate orders into "Pools" to minimize logistics cost.
    Input: List of separate orders.
    Output: Optimized Pools (groups of neighbors).
    """
    pools = []
    processed = set()
    pool_counter = 1

    # Loop through every order
    for i, order in enumerate(orders):
        if order.shop_id in processed:
            continue
        
        # Start a new pool with this shop as the "Anchor"
        current_pool = {
            "pool_id": f"POOL-{pool_counter:03d}",
            "shops": [order.shop_id],
            "total_qty": order.qty_needed,
            "center_lat": order.lat,
            "center_lon": order.lon,
            "radius_km": 0.0
        }
        processed.add(order.shop_id)

        # Look for neighbors to merge into this pool
        for j, neighbor in enumerate(orders):
            if neighbor.shop_id in processed:
                continue
            
            # Check Distance
            dist = calculate_distance(order.lat, order.lon, neighbor.lat, neighbor.lon)
            
            # 3 KM Clumping Radius
            if dist < 3.0: 
                current_pool["shops"].append(neighbor.shop_id)
                current_pool["total_qty"] += neighbor.qty_needed
                # Update radius to the furthest point in the group
                if dist > current_pool["radius_km"]:
                    current_pool["radius_km"] = round(dist, 2)
                processed.add(neighbor.shop_id)
        
        # Add 'Bulk Discount' Tag if pool is large
        if current_pool["total_qty"] > 50:
            current_pool["discount"] = "15% WHOLESALE"
            current_pool["discount_val"] = 0.15
        else:
            current_pool["discount"] = "STANDARD"
            current_pool["discount_val"] = 0.0
            
        pools.append(current_pool)
        pool_counter += 1

    return pools

# ==========================================
# ENDPOINT 2: RETAILER RECOMMENDER SYSTEM
# ==========================================
@app.post("/recommend_distributor")
def recommend_distributor(shop: RetailerRequest):
    """
    Suggests the best distributor based on Shop's Urgency (Risk).
    """
    # 1. Ask AI for Risk Score (Simulated Logic)
    risk_score = 0.2
    if shop.is_festival: risk_score += 0.4
    if shop.current_stock < 15: risk_score += 0.3
    
    # 2. Define Available Distributors (Mock Database)
    distributors = [
        {"name": "FastTrack Logistics", "cost": 100, "speed_hrs": 4, "reliability": 0.98},
        {"name": "Budget Movers",       "cost": 60,  "speed_hrs": 24, "reliability": 0.85},
        {"name": "GraminRoute Hub",     "cost": 75,  "speed_hrs": 12, "reliability": 0.99} 
    ]
    
    # 3. Intelligent Ranking Logic
    recommendations = []
    
    for dist in distributors:
        score = 0
        festival_boost = 1.5 if shop.is_festival else 1.0

        # SCENARIO A: HIGH RISK (stock out soon)
        if risk_score > 0.7:
            # Prioritize speed, Reliability matters a lot
            score += (100 / dist["speed_hrs"]) * 2.0 * festival_boost 
            score += (dist["reliability"] * 120) * festival_boost     
            reason = "⚡ FASTEST (Urgent Need)"

        # SCENARIO B: FESTIVE BULK ORDER (Stable but high demand)
        elif shop.is_festival and risk_score <= 0.7:
            # Reliability + Volume cost
            score += (dist["reliability"] * 150)
            score += (50 / dist["speed_hrs"]) * 3.0 
            if dist["cost"] < 80:
                score += 40 
            reason = "🎉 FESTIVAL WHOLESALE"

        # SCENARIO C: Routine order
        else:
            # Cost is king
            score += (200 / dist["cost"]) * 2.0      
            score += (dist["reliability"] * 50)
            reason = "💰 BEST PRICE (Routine)"

        recommendations.append({
            "distributor": dist["name"],
            "match_score": round(score, 1),
            "reason": reason,
            "cost": dist["cost"],
            "eta": f"{dist['speed_hrs']} Hours"
        })
        
    # Sort by Score (Highest First)
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    
    return {
        "shop_status": "CRITICAL" if risk_score > 0.7 else "STABLE",
        "top_pick": recommendations[0],
        "all_options": recommendations
    }