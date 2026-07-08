import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from torch_geometric.data import Data

# Make sure backend/ is on sys.path so ml/ and services/ resolve correctly.
# Run from the backend/ directory: uvicorn api.main:app --reload
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.model_def         import SpatialGNN
from ml.risk_model        import get_risk_model, build_features
from ml.festival_predictor import compute_stockout_forecast
from ml.recommender       import get_recommender
from services.pooling     import generate_pools, normalise_orders
from api.schemas          import RetailerInput, PendingOrder

# ==========================================
# APP
# ==========================================

app = FastAPI(title="GraminRoute API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://gramin-route1-kpj2q6g19-vishals-projects-8bf76249.vercel.app",
        "https://gramin-route1-r83apuouf-vishals-projects-8bf76249.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MODEL LOADING
# ==========================================

BASE_DIR = Path(__file__).parent.parent          # backend/
MODELS   = BASE_DIR / "models"

# 1. XGBoost Risk Model
risk_model = get_risk_model()

# 2. SpatialGNN  (in_dim=10: 9 features + XGBoost risk score)
spatial_gnn = SpatialGNN(in_dim=10, hidden_dim=64)
_gnn_weights = MODELS / "spatial_gnn.pth"
try:
    spatial_gnn.load_state_dict(torch.load(_gnn_weights, map_location="cpu"))
    spatial_gnn.eval()
    print("✅ SpatialGNN weights loaded.")
except Exception as e:
    print(f"⚠️  SpatialGNN weights not found ({e}). Running untrained — will retrain.")
    spatial_gnn.eval()

# 3. XGBoost Distributor Recommender
recommender = get_recommender()

print("✅ GraminRoute v2 — all models ready.")

# ==========================================
# HELPERS
# ==========================================

def run_spatial_gnn(features_9: np.ndarray, xgb_risk: float) -> float:
    """
    Appends XGBoost risk score to features → 10 features.
    Uses a self-loop graph for single-node inference.
    In production, all village nodes run together as a full district graph.
    """
    x          = torch.tensor(np.append(features_9, xgb_risk), dtype=torch.float32).unsqueeze(0)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_attr  = torch.tensor([[1.0]], dtype=torch.float32)

    with torch.no_grad():
        output = spatial_gnn(Data(x=x, edge_index=edge_index, edge_attr=edge_attr)).item()

    return float(np.clip(output, 0.0, 1.0))


def shop_status(risk: float) -> str:
    return "CRITICAL" if risk > 0.7 else "WARNING" if risk > 0.4 else "STABLE"


# ==========================================
# ENDPOINT 1: DISTRIBUTOR RECOMMENDER
# ==========================================

@app.post("/recommend_distributor")
def recommend_distributor(shop: RetailerInput):
    """
    Full 5-stage pipeline:
      Feature Engineering
        → XGBoost Risk  (individual, no graph)
        → SpatialGNN    (road-network spatial context)
        → Festival Predictor  (stockout forecast)
        → XGBoost Recommender  (ranked distributor list)
    """
    # Stage 1 — features
    features_9, festival_ctx = build_features(
        current_stock  = shop.current_stock,
        daily_sales    = shop.daily_sales,
        lead_time_days = shop.lead_time_days,
        profit_margin  = shop.profit_margin,
        shelf_life     = shop.shelf_life,
        credit_score   = shop.credit_score,
        product_name   = shop.product_name,
    )

    # Stage 2 — XGBoost individual risk
    xgb_risk = risk_model.predict_risk(features_9)

    # Stage 3 — SpatialGNN spatial risk
    spatial_risk = run_spatial_gnn(features_9, xgb_risk)

    # Stage 4 — Festival stockout forecast
    forecast = compute_stockout_forecast(
        current_stock  = shop.current_stock,
        daily_sales    = shop.daily_sales,
        festival_ctx   = festival_ctx,
        lead_time_days = shop.lead_time_days,
    )

    # Stage 5 — Distributor ranking
    ranked = recommender.rank(
        spatial_risk_score  = spatial_risk,
        days_until_stockout = forecast["days_until_stockout"],
        days_to_festival    = forecast["days_to_festival"],
        credit_score        = shop.credit_score,
        recommended_qty     = forecast["recommended_order_qty"],
    )

    return {
        "shop_id":             shop.shop_id,
        "xgb_risk_score":      round(xgb_risk, 2),
        "spatial_risk_score":  round(spatial_risk, 2),
        "shop_status":         shop_status(spatial_risk),
        "days_until_stockout": forecast["days_until_stockout"],
        "restock_urgency":     forecast["restock_urgency"],
        "recommended_order_qty": forecast["recommended_order_qty"],
        "restock_deadline":    forecast["restock_deadline"],
        "festival_alert": {
            "festival_name":    festival_ctx["festival_name"],
            "days_away":        festival_ctx["days_to_festival"],
            "demand_multiplier": forecast["demand_multiplier"],
            "affected_products": festival_ctx["affected_products"],
            "in_prep_window":   festival_ctx["in_prep_window"],
        },
        "top_pick":             ranked[0],
        "alternatives":         ranked[1:],
        "feature_importance":   risk_model.feature_importance(),
    }


# ==========================================
# ENDPOINT 2: POOLING ENGINE
# ==========================================

@app.post("/pool_orders")
def pool_orders(orders: List[PendingOrder]):
    """
    DBSCAN spatial clustering (3km radius, Haversine metric).
    Shops labelled -1 (isolated) get solo delivery.
    """
    clean = normalise_orders(orders)
    return generate_pools(clean)


# ==========================================
# ENDPOINT 3: FINANCIAL SIMULATION
# ==========================================

@app.get("/simulate_savings")
def simulate_savings():
    """60-day simulation: Traditional vs GraminRoute (XGBoost-driven)."""
    std_cash, std_stock = 50000.0, 50
    ai_cash,  ai_stock  = 50000.0, 50
    base_cost, bulk_cost, delivery_fee, pooled_fee, margin = 80.0, 75.0, 100.0, 25.0, 20.0
    history = []

    for day in range(1, 61):
        is_festival  = 20 <= day <= 25
        daily_demand = random.randint(2, 8) + (10 if is_festival else 0)

        # Traditional
        if std_stock < 20:
            std_cash -= 40 * base_cost + delivery_fee
            std_stock += 40

        # GraminRoute — XGBoost decides target stock
        features_9 = np.array([
            ai_stock / 200.0, daily_demand / 50.0, 3.0 / 7.0,
            20.0 / 60.0, 0.08, 0.78,
            0.17 if is_festival else 0.5,
            (2.8 if is_festival else 1.0) / 3.0,
            1.0 if is_festival else 0.0,
        ], dtype=np.float32)

        xgb_risk = risk_model.predict_risk(features_9)
        target   = 60 if xgb_risk > 0.6 else 40 if xgb_risk > 0.4 else 15

        if ai_stock < target:
            needed = target - ai_stock
            pooled = needed >= 40 or random.random() > 0.3
            ai_cash  -= (needed * bulk_cost + pooled_fee) if pooled else (needed * base_cost + delivery_fee)
            ai_stock += needed

        # Sales
        std_sold, ai_sold = min(std_stock, daily_demand), min(ai_stock, daily_demand)
        std_stock -= std_sold; std_cash += std_sold * (base_cost + margin)
        ai_stock  -= ai_sold;  ai_cash  += ai_sold  * (base_cost + margin)

        history.append({
            "day":         f"Day {day}",
            "Traditional": round(std_cash + std_stock * base_cost),
            "GraminRoute": round(ai_cash  + ai_stock  * base_cost),
            "isFestival":  is_festival,
        })

    return history


# ==========================================
# ENDPOINT 4: MODEL INTERPRETABILITY
# ==========================================

@app.get("/model_info")
def model_info():
    return {
        "risk_model": {
            "type":     "XGBoost Classifier (9 features)",
            "features": risk_model.feature_importance(),
            "note":     "Per-shop risk. Ignores graph/spatial context.",
        },
        "spatial_gnn": {
            "type":      "GATv2 Graph Attention Network",
            "layers":    "2 × GATv2Conv (4 heads → 1 head)",
            "in_dim":    10,
            "hidden_dim": 64,
            "note":      "Propagates risk through village road network.",
        },
        "recommender": {
            "type":    "XGBoost Multi-class (Contextual Bandit framing)",
            "classes": ["FastTrack Logistics", "GraminRoute Hub", "Budget Movers"],
            "note":    "Ranks distributors by confidence score.",
        },
    }
