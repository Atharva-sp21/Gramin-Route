from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import random
from typing import List, Optional
from math import radians, sin, cos, sqrt, atan2
from torch_geometric.data import Data

# Import the Quantum Model Architecture
# Ensure backend/model_def.py exists with the class definition provided previously
from model_def import HybridQuantumGNN 

# ... imports ...

app = FastAPI()

# --- 0. CORS CONFIGURATION (FIXED) ---
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... rest of your code ...
# --- 1. LOAD THE TRAINED QUANTUM BRAIN ---
model = HybridQuantumGNN(in_dim=7)
MODEL_PATH = "model/quantum_gnn_model.pth"

try:
    # Load weights onto CPU (since Render/Vercel usually don't have GPUs)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Quantum Brain Loaded & Ready for B2B Inference")
except Exception as e:
    print(f"⚠️ CRITICAL: Model load failed ({e}). System running in Fallback Mode.")
    model = None

# --- 2. DATA MODELS ---

class RetailerInput(BaseModel):
    # UPDATE: Changed 'id' to 'shop_id' to match the frontend JSON
    shop_id: str 
    lat: float
    lon: float
    current_stock: int
    daily_sales: int = 5         
    lead_time_days: int = 3      
    profit_margin: float = 20.0  
    shelf_life: int = 30         
    is_festival: bool = False
    credit_score: int = 700

class PendingOrder(BaseModel):
    shop_id: str
    lat: float
    lon: float
    qty_needed: int
    retailer_id: Optional[str] = None # Handle diverse naming conventions
    retailer_lat: Optional[float] = None
    retailer_lon: Optional[float] = None

# --- 3. HELPER: NORMALIZATION (Crucial for AI Accuracy) ---
def normalize_features(data: RetailerInput):
    """
    Matches the normalization logic from the Jupyter Notebook (Block 2).
    Maps raw business data to the 0-1 range the Quantum Model expects.
    """
    f1 = data.current_stock / 200.0
    f2 = data.daily_sales / 50.0
    f3 = data.lead_time_days / 7.0
    f4 = data.profit_margin / 60.0
    f5 = data.shelf_life / 365.0
    f6 = 1.0 if data.is_festival else 0.0
    f7 = data.credit_score / 900.0
    
    # Create Tensor: Shape [1, 7]
    return torch.tensor([[f1, f2, f3, f4, f5, f6, f7]], dtype=torch.float32)

def get_risk_prediction(features):
    """
    Runs the model. 
    Since GNNs need a graph, we create a 'dummy' self-loop graph for single-node inference.
    """
    if not model: return 0.5 # Fallback
    
    # Dummy Edge Index (Self-loop)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    # Dummy Edge Attr (Distance=1.0, RoadType=1.0)
    edge_attr = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32) 
    
    # --- FIX: Bundle everything into a Data object ---
    data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)
    
    with torch.no_grad():
        # Pass the single 'data' object instead of 3 separate arguments
        output = model(data).item()
        
    return max(0.0, min(1.0, output))

# ==========================================
# ENDPOINT 1: B2B DISTRIBUTOR RECOMMENDER
# ==========================================
# ==========================================
# ENDPOINT 1: B2B DISTRIBUTOR RECOMMENDER
# ==========================================
@app.post("/recommend_distributor")
def recommend_distributor(shop: RetailerInput):
    """
    Input: Retailer Details.
    Process: AI predicts 'Late Delivery Risk'.
    Output: Best Distributor + Business Reason.
    """
    # 1. Run AI Inference
    features = normalize_features(shop)
    risk_score = get_risk_prediction(features)
    
    # 2. Define Distributor Tier List
    distributors = [
        {"name": "FastTrack Logistics", "cost": 100, "speed": 4, "rel": 0.99, "type": "PREMIUM"},
        {"name": "GraminRoute Hub",     "cost": 75,  "speed": 12, "rel": 0.95, "type": "BALANCED"},
        {"name": "Budget Movers",       "cost": 60,  "speed": 24, "rel": 0.85, "type": "ECONOMY"}
    ]
    
    recommendations = []
    status = "STABLE"
    
    # 3. Decision Matrix
    for dist in distributors:
        score = 0
        
        # Scenario A: HIGH RISK (Risk > 70% OR Festival)
        if risk_score > 0.7 or shop.is_festival:
            status = "CRITICAL"
            # Prioritize Speed & Reliability above all
            score += (10 / dist["speed"]) * 50 
            score += (dist["rel"] * 100)
            reason = f"⚡ URGENT: High AI Risk Score ({int(risk_score*100)}%)"
            
        # Scenario B: MEDIUM RISK (Low Stock)
        elif shop.current_stock < 20:
            status = "WARNING"
            score += (dist["rel"] * 80)
            score += (100 / dist["cost"]) * 20
            reason = "📉 LOW STOCK: Balanced Re-stock Needed"
            
        # Scenario C: LOW RISK (Routine)
        else:
            status = "STABLE"
            # Prioritize Cost Savings
            score += (100 / dist["cost"]) * 50
            score += (dist["rel"] * 10)
            reason = "💰 COST SAVING: Low Risk Environment"
            
        recommendations.append({
            "distributor": dist["name"],
            "match_score": round(score, 1),
            "reason": reason,
            "cost": dist["cost"],
            "eta": f"{dist['speed']} Hours"
        })
        
    # Sort by highest match
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    
    return {
        # UPDATE: Accessed via shop.shop_id instead of shop.id
        "shop_id": shop.shop_id,
        "ai_risk_score": round(risk_score, 2),
        "shop_status": status,
        "top_pick": recommendations[0],
        "alternatives": recommendations[1:]
    }

# ==========================================
# ENDPOINT 2: GEOSPATIAL POOLING ENGINE
# ==========================================
@app.post("/pool_orders")
def generate_pools(orders: List[PendingOrder]):
    """
    Groups individual orders into pools based on 3KM radius.
    """
    pools = []
    processed = set()
    pool_counter = 1

    # Normalize input data (Handle different naming conventions from frontend)
    clean_orders = []
    for o in orders:
        clean_orders.append({
            "id": o.shop_id,
            "qty": o.qty_needed,
            "lat": o.retailer_lat if o.retailer_lat else o.lat,
            "lon": o.retailer_lon if o.retailer_lon else o.lon
        })

    for i, order in enumerate(clean_orders):
        if order["id"] in processed: continue
        
        # Start Pool
        current_pool = {
            "pool_id": f"POOL-{pool_counter:03d}",
            "shops": [order["id"]],
            "total_qty": order["qty"],
            "center_lat": order["lat"],
            "center_lon": order["lon"],
            "radius_km": 0.0
        }
        processed.add(order["id"])

        # Find Neighbors
        for j, neighbor in enumerate(clean_orders):
            if neighbor["id"] in processed: continue
            
            # Haversine Distance
            R = 6371.0
            dlat = radians(neighbor["lat"] - order["lat"])
            dlon = radians(neighbor["lon"] - order["lon"])
            a = sin(dlat/2)**2 + cos(radians(order["lat"])) * cos(radians(neighbor["lat"])) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            dist = R * c
            
            # 3KM Clustering Logic
            if dist < 3.0: 
                current_pool["shops"].append(neighbor["id"])
                current_pool["total_qty"] += neighbor["qty"]
                current_pool["radius_km"] = max(current_pool["radius_km"], dist)
                processed.add(neighbor["id"])
        
        # Apply Discounts
        if current_pool["total_qty"] > 50:
            current_pool["discount"] = "15% WHOLESALE"
        else:
            current_pool["discount"] = "STANDARD"
            
        pools.append(current_pool)
        pool_counter += 1

    return pools

# ==========================================
# ENDPOINT 3: FINANCIAL SIMULATION (Using Model Inference)
# ==========================================
@app.get("/simulate_savings")
def get_simulation():
    """
    Runs a 60-day simulation where the AI decides re-stocking 
    based on the Model's Risk Score vs Traditional Logic.
    """
    days = 60
    history = []
    
    # State
    std_cash, std_stock = 50000.0, 50
    ai_cash, ai_stock = 50000.0, 50
    
    # Costs
    base_cost = 80.0
    bulk_cost = 75.0
    delivery_fee = 100.0
    pooled_delivery = 25.0
    margin = 20.0

    for day in range(1, days + 1):
        # 1. Market Context
        is_festival = (20 <= day <= 25)
        daily_demand = random.randint(2, 8)
        if is_festival: daily_demand += 10
        
        # --- TRADITIONAL (Reactive) ---
        if std_stock < 20:
            order_qty = 40
            std_cash -= (order_qty * base_cost) + delivery_fee
            std_stock += order_qty

        # --- GRAMINROUTE AI (Predictive via Model) ---
        target_stock = 15 
        
        if model:
            # Create feature vector for TODAY'S state
            # Notice we pass the dynamic 'ai_stock' and 'daily_demand' into the model
            features = torch.tensor([[
                ai_stock / 200.0,
                daily_demand / 50.0,
                3.0 / 7.0,   # Lead time constant
                20.0 / 60.0, # Margin constant
                0.08,        # Shelf life constant
                1.0 if is_festival else 0.0,
                0.78         # Good credit score
            ]], dtype=torch.float32)

            risk_score = get_risk_prediction(features)
            
            # AI interprets Risk Score
            if risk_score > 0.6: target_stock = 60 # Overstock
            elif risk_score > 0.4: target_stock = 40
            else: target_stock = 15
        else:
            target_stock = 50 if is_festival else 15

        # Execute AI Order
        if ai_stock < target_stock:
            needed = target_stock - ai_stock
            is_pooled = (needed >= 40) or (random.random() > 0.3)
            cost = (needed * bulk_cost) + pooled_delivery if is_pooled else (needed * base_cost) + delivery_fee
            ai_cash -= cost
            ai_stock += needed

        # --- SALES ---
        std_sold = min(std_stock, daily_demand)
        std_stock -= std_sold
        std_cash += std_sold * (base_cost + margin)
        
        ai_sold = min(ai_stock, daily_demand)
        ai_stock -= ai_sold
        ai_cash += ai_sold * (base_cost + margin)
        
        history.append({
            "day": f"Day {day}",
            "Traditional": round(std_cash + (std_stock * base_cost)),
            "GraminRoute": round(ai_cash + (ai_stock * base_cost)),
            "isFestival": is_festival
        })
        
    return history