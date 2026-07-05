from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import random
from typing import List, Optional
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
from torch_geometric.data import Data

# --- New modular pipeline imports ---
from model_def import SpatialGNN
from risk_model import get_risk_model, build_features
from festival_predictor import compute_stockout_forecast
from recommender import get_recommender

app = FastAPI(title="GraminRoute API", version="2.0")

# ==========================================
# 0. CORS
# ==========================================
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://gramin-route1-kpj2q6g19-vishals-projects-8bf76249.vercel.app",
    "https://gramin-route1-r83apuouf-vishals-projects-8bf76249.vercel.app/",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD MODELS (all three on startup)
# ==========================================

# 1a. XGBoost Risk Model (trains on synthetic data if no .pkl found)
risk_model = get_risk_model()

# 1b. SpatialGNN — GATv2 without quantum (in_dim=10: 9 features + XGB risk score)
SPATIAL_GNN_PATH = "model/spatial_gnn.pth"
spatial_gnn = SpatialGNN(in_dim=10, hidden_dim=64)
try:
    spatial_gnn.load_state_dict(
        torch.load(SPATIAL_GNN_PATH, map_location="cpu")
    )
    spatial_gnn.eval()
    print("✅ SpatialGNN loaded.")
except Exception as e:
    print(f"⚠️  SpatialGNN weights not found ({e}). Running in untrained mode.")
    spatial_gnn.eval()

# 1c. XGBoost Distributor Recommender
recommender = get_recommender()

print("✅ GraminRoute v2 — all models ready.")

# ==========================================
# 2. INPUT SCHEMAS
# ==========================================

class RetailerInput(BaseModel):
    shop_id: str
    lat: float
    lon: float
    current_stock: int
    daily_sales: int = 5
    lead_time_days: int = 3
    profit_margin: float = 20.0
    shelf_life: int = 30
    credit_score: int = 700
    # Festival fields — auto-derived from calendar if not provided
    product_name: str = "Rice (50kg)"   # Used to look up festival affinity


class PendingOrder(BaseModel):
    shop_id: str
    lat: float
    lon: float
    qty_needed: int
    retailer_id: Optional[str] = None
    retailer_lat: Optional[float] = None
    retailer_lon: Optional[float] = None


# ==========================================
# 3. INFERENCE HELPERS
# ==========================================

def run_spatial_gnn(features_9: np.ndarray, xgb_risk: float) -> float:
    """
    Appends the XGBoost risk score to the 9-feature vector,
    creates a self-loop graph for single-node inference,
    and runs SpatialGNN.

    In production this would receive ALL village nodes at once
    (a full district graph). For single-shop API calls we use
    a self-loop as a minimal valid graph — the GATv2 attention
    degenerates to a per-node transform, which is still useful
    as a learned non-linear projection.
    """
    # Append XGB risk score → 10 features
    features_10 = np.append(features_9, xgb_risk).astype(np.float32)
    x = torch.tensor(features_10, dtype=torch.float32).unsqueeze(0)  # [1, 10]

    # Minimal self-loop graph (single node inference)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_attr  = torch.tensor([[1.0]], dtype=torch.float32)  # highway weight

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    with torch.no_grad():
        output = spatial_gnn(data).item()

    return float(np.clip(output, 0.0, 1.0))


def shop_status(risk: float) -> str:
    if risk > 0.7:  return "CRITICAL"
    if risk > 0.4:  return "WARNING"
    return "STABLE"


# ==========================================
# ENDPOINT 1: B2B DISTRIBUTOR RECOMMENDER
# ==========================================

@app.post("/recommend_distributor")
def recommend_distributor(shop: RetailerInput):
    """
    Full pipeline:
        raw features
          → XGBoost risk score (per-shop, ignores neighbors)
          → SpatialGNN (adds road-network spatial context)
          → Festival Predictor (days until stockout, recommended qty)
          → XGBoost Recommender (ranked distributor list)

    Returns enriched response vs. v1:
        + days_until_stockout
        + festival_alert  (festival name, days away, affected products)
        + recommended_order_qty
        + restock_deadline
        + distributor confidence scores (vs. opaque match_score)
    """
    # Step 1: Build 9-feature vector + festival context
    features_9, festival_ctx = build_features(
        current_stock  = shop.current_stock,
        daily_sales    = shop.daily_sales,
        lead_time_days = shop.lead_time_days,
        profit_margin  = shop.profit_margin,
        shelf_life     = shop.shelf_life,
        credit_score   = shop.credit_score,
        product_name   = shop.product_name,
    )

    # Step 2: XGBoost — individual risk score (no graph)
    xgb_risk = risk_model.predict_risk(features_9)

    # Step 3: SpatialGNN — add road-network neighborhood context
    spatial_risk = run_spatial_gnn(features_9, xgb_risk)

    # Step 4: Festival predictor — stockout forecast
    forecast = compute_stockout_forecast(
        current_stock  = shop.current_stock,
        daily_sales    = shop.daily_sales,
        festival_ctx   = festival_ctx,
        lead_time_days = shop.lead_time_days,
    )

    # Step 5: Recommender — rank distributors
    ranked = recommender.rank(
        spatial_risk_score  = spatial_risk,
        days_until_stockout = forecast["days_until_stockout"],
        days_to_festival    = forecast["days_to_festival"],
        credit_score        = shop.credit_score,
        recommended_qty     = forecast["recommended_order_qty"],
    )

    return {
        "shop_id":              shop.shop_id,
        "xgb_risk_score":       round(xgb_risk, 2),
        "spatial_risk_score":   round(spatial_risk, 2),
        "shop_status":          shop_status(spatial_risk),

        # Stockout forecast (new in v2)
        "days_until_stockout":  forecast["days_until_stockout"],
        "restock_urgency":      forecast["restock_urgency"],
        "recommended_order_qty": forecast["recommended_order_qty"],
        "restock_deadline":     forecast["restock_deadline"],

        # Festival alert (new in v2)
        "festival_alert": {
            "festival_name":    festival_ctx["festival_name"],
            "days_away":        festival_ctx["days_to_festival"],
            "demand_multiplier": forecast["demand_multiplier"],
            "affected_products": festival_ctx["affected_products"],
            "in_prep_window":   festival_ctx["in_prep_window"],
        },

        # Distributor ranking (confidence scores instead of opaque match_score)
        "top_pick":    ranked[0],
        "alternatives": ranked[1:],

        # XGBoost interpretability (new in v2)
        "feature_importance": risk_model.feature_importance(),
    }


# ==========================================
# ENDPOINT 2: GEOSPATIAL POOLING ENGINE (DBSCAN)
# ==========================================

@app.post("/pool_orders")
def generate_pools(orders: List[PendingOrder]):
    """
    Groups orders into delivery pools using DBSCAN (Haversine metric).
    Replaces the manual greedy Haversine loop from v1.

    eps = 3km radius (converted to radians for sklearn Haversine).
    Shops labelled -1 (noise) get individual delivery — correct
    behaviour for isolated rural shops with no nearby orders.
    """
    if not orders:
        return []

    # Normalise lat/lon from both naming conventions
    clean = []
    for o in orders:
        clean.append({
            "id":  o.shop_id or o.retailer_id,
            "qty": o.qty_needed,
            "lat": o.retailer_lat if o.retailer_lat else o.lat,
            "lon": o.retailer_lon if o.retailer_lon else o.lon,
        })

    coords = np.radians([[c["lat"], c["lon"]] for c in clean])

    # DBSCAN with Haversine — eps in radians (3km / Earth radius)
    db = DBSCAN(
        eps=3.0 / 6371.0,
        min_samples=1,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords)

    labels = db.labels_
    unique_labels = set(labels)

    pools = []
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        pool_shops = [clean[i] for i in indices]

        total_qty = sum(s["qty"] for s in pool_shops)
        center_lat = np.mean([s["lat"] for s in pool_shops])
        center_lon = np.mean([s["lon"] for s in pool_shops])

        # Max radius within the pool
        max_radius = 0.0
        for s in pool_shops:
            R = 6371.0
            dlat = radians(s["lat"] - center_lat)
            dlon = radians(s["lon"] - center_lon)
            a = (sin(dlat/2)**2
                 + cos(radians(center_lat)) * cos(radians(s["lat"])) * sin(dlon/2)**2)
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


# ==========================================
# ENDPOINT 3: FINANCIAL SIMULATION
# ==========================================

@app.get("/simulate_savings")
def get_simulation():
    """
    60-day simulation: Traditional (reactive) vs GraminRoute (predictive).
    Uses XGBoost risk score instead of the old quantum model output.
    """
    days = 60
    history = []

    std_cash, std_stock = 50000.0, 50
    ai_cash,  ai_stock  = 50000.0, 50

    base_cost    = 80.0
    bulk_cost    = 75.0
    delivery_fee = 100.0
    pooled_fee   = 25.0
    margin       = 20.0

    for day in range(1, days + 1):
        is_festival  = (20 <= day <= 25)
        daily_demand = random.randint(2, 8) + (10 if is_festival else 0)

        # --- Traditional (reactive) ---
        if std_stock < 20:
            qty = 40
            std_cash  -= (qty * base_cost) + delivery_fee
            std_stock += qty

        # --- GraminRoute (predictive via XGBoost) ---
        features_9 = np.array([
            ai_stock / 200.0,
            daily_demand / 50.0,
            3.0 / 7.0,
            20.0 / 60.0,
            0.08,
            0.78,
            0.17 if is_festival else 0.5,   # days_to_festival normalized
            (2.8 if is_festival else 1.0) / 3.0,  # spike factor
            1.0 if is_festival else 0.0,    # product affinity
        ], dtype=np.float32)

        xgb_risk  = risk_model.predict_risk(features_9)
        target    = 60 if xgb_risk > 0.6 else (40 if xgb_risk > 0.4 else 15)

        if ai_stock < target:
            needed = target - ai_stock
            pooled = (needed >= 40) or (random.random() > 0.3)
            cost   = (needed * bulk_cost + pooled_fee) if pooled \
                     else (needed * base_cost + delivery_fee)
            ai_cash  -= cost
            ai_stock += needed

        # --- Sales ---
        std_sold  = min(std_stock, daily_demand)
        std_stock -= std_sold
        std_cash  += std_sold * (base_cost + margin)

        ai_sold  = min(ai_stock, daily_demand)
        ai_stock -= ai_sold
        ai_cash  += ai_sold * (base_cost + margin)

        history.append({
            "day":         f"Day {day}",
            "Traditional": round(std_cash + std_stock * base_cost),
            "GraminRoute": round(ai_cash  + ai_stock  * base_cost),
            "isFestival":  is_festival,
        })

    return history


# ==========================================
# ENDPOINT 4: MODEL INTERPRETABILITY (new in v2)
# ==========================================

@app.get("/model_info")
def model_info():
    """
    Returns XGBoost feature importances for both models.
    Useful for demo / interview — shows which features
    drive risk and distributor selection.
    """
    return {
        "risk_model": {
            "type": "XGBoost Classifier",
            "features": risk_model.feature_importance(),
            "description": "Predicts stockout risk per shop, ignoring spatial context.",
        },
        "spatial_gnn": {
            "type": "GATv2 Graph Attention Network",
            "layers": "2 GATv2Conv (4 heads, 1 head)",
            "in_dim": 10,
            "hidden_dim": 64,
            "description": "Propagates risk signals through the village road network.",
        },
        "recommender": {
            "type": "XGBoost Multi-class Classifier (Contextual Bandit framing)",
            "classes": ["FastTrack Logistics", "GraminRoute Hub", "Budget Movers"],
            "description": "Ranks distributors by context-conditioned confidence score.",
        },
    }
