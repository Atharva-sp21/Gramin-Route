"""
GraminRoute — Analytics / ML Service
======================================
Runs in the background, NOT in the critical path... except for
/recommend_distributor, which is a synchronous request/response
pipeline ported over from the old standalone backend/ (retired).

- Consumes inventory.updated to keep running demand stats (EWA in Redis)
- Exposes /model_info and /simulate_savings for the frontend
- Exposes /recommend_distributor — the full 5-stage AI pipeline:
    Feature Engineering → XGBoost Risk → SpatialGNN →
    Festival Forecast → XGBoost Distributor Recommender
- Future: Airflow nightly LightGBM demand forecast
"""

import os
import json
import logging
import random
import asyncio
from pathlib import Path

import numpy as np
import torch
import asyncpg
import redis.asyncio as aioredis
from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.data import Data

from risk import get_risk_model, build_features
from model_def import SpatialGNN
from festival_predictor import compute_stockout_forecast
from recommender import get_recommender

# ── Config ───────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gramin:gramin@localhost:5432/graminroute")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
KAFKA_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ml_service")

# ── FastAPI app (for model_info, simulate_savings) ─
app = FastAPI(title="GraminRoute ML Service", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_conn = None

# ── Model loading (recommend_distributor pipeline) ─
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# 1. XGBoost Risk Model
risk_model = get_risk_model()

# 2. SpatialGNN  (in_dim=10: 9 features + XGBoost risk score)
spatial_gnn = SpatialGNN(in_dim=10, hidden_dim=64)
_gnn_weights = MODELS_DIR / "spatial_gnn.pth"
try:
    spatial_gnn.load_state_dict(torch.load(_gnn_weights, map_location="cpu"))
    spatial_gnn.eval()
    log.info("✅ SpatialGNN weights loaded.")
except Exception as e:
    log.warning(f"⚠️  SpatialGNN weights not found ({e}). Running untrained.")
    spatial_gnn.eval()

# 3. XGBoost Distributor Recommender
recommender = get_recommender()


def run_spatial_gnn(features_9: np.ndarray, xgb_risk: float) -> float:
    """
    Appends XGBoost risk score to features → 10 features.
    Uses a self-loop graph for single-node inference.
    In production, all village nodes run together as a full district graph.
    """
    x = torch.tensor(np.append(features_9, xgb_risk), dtype=torch.float32).unsqueeze(0)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0]], dtype=torch.float32)

    with torch.no_grad():
        output = spatial_gnn(Data(x=x, edge_index=edge_index, edge_attr=edge_attr)).item()

    return float(np.clip(output, 0.0, 1.0))


def shop_status(risk: float) -> str:
    return "CRITICAL" if risk > 0.7 else "WARNING" if risk > 0.4 else "STABLE"


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
    # Product name drives festival calendar lookup
    product_name: str = "Rice (50kg)"


@app.on_event("startup")
async def startup():
    global redis_conn
    redis_conn = aioredis.from_url(REDIS_URL, decode_responses=True)
    # Start Kafka consumer in background
    asyncio.create_task(consume_inventory_events())
    log.info("✅ ML Service started")


@app.get("/model_info")
async def model_info():
    """Model metadata for frontend transparency."""
    return {
        "risk_model": {
            "type": "XGBoost Classifier (9 features)",
            "features": [
                "stock_ratio", "sales_ratio", "lead_time_ratio",
                "margin_ratio", "shelf_life_ratio", "credit_ratio",
                "days_to_festival", "spike_factor", "product_affinity",
            ],
            "note": "Per-shop risk. Runs in Inventory Service on every stock update.",
        },
        "spatial_gnn": {
            "type": "GATv2 Graph Attention Network",
            "layers": "2 × GATv2Conv (4 heads → 1 head)",
            "in_dim": 10,
            "hidden_dim": 64,
            "note": "Propagates risk through village road network. Future: runs nightly.",
        },
        "pool_formation": {
            "type": "PostGIS ST_DWithin (2km radius)",
            "note": "Replaces DBSCAN. Uses real road network geometry.",
        },
        "route_optimizer": {
            "type": "Nearest-Neighbor TSP",
            "note": "Greedy solver, optimal for 4-8 stops.",
        },
    }


@app.post("/recommend_distributor")
async def recommend_distributor(shop: RetailerInput):
    """
    Full 5-stage pipeline (ported from the retired standalone backend/):
      Feature Engineering
        → XGBoost Risk  (individual, no graph)
        → SpatialGNN    (road-network spatial context)
        → Festival Predictor  (stockout forecast)
        → XGBoost Recommender  (ranked distributor list)
    """
    # Stage 1 — features
    features_9, festival_ctx = build_features(
        current_stock=shop.current_stock,
        daily_sales=shop.daily_sales,
        lead_time_days=shop.lead_time_days,
        profit_margin=shop.profit_margin,
        shelf_life=shop.shelf_life,
        credit_score=shop.credit_score,
        product_name=shop.product_name,
    )

    # Stage 2 — XGBoost individual risk
    xgb_risk = risk_model.predict_risk(features_9)

    # Stage 3 — SpatialGNN spatial risk
    spatial_risk = run_spatial_gnn(features_9, xgb_risk)

    # Stage 4 — Festival stockout forecast
    forecast = compute_stockout_forecast(
        current_stock=shop.current_stock,
        daily_sales=shop.daily_sales,
        festival_ctx=festival_ctx,
        lead_time_days=shop.lead_time_days,
    )

    # Stage 5 — Distributor ranking
    ranked = recommender.rank(
        spatial_risk_score=spatial_risk,
        days_until_stockout=forecast["days_until_stockout"],
        days_to_festival=forecast["days_to_festival"],
        credit_score=shop.credit_score,
        recommended_qty=forecast["recommended_order_qty"],
    )

    return {
        "shop_id": shop.shop_id,
        "xgb_risk_score": round(xgb_risk, 2),
        "spatial_risk_score": round(spatial_risk, 2),
        "shop_status": shop_status(spatial_risk),
        "days_until_stockout": forecast["days_until_stockout"],
        "restock_urgency": forecast["restock_urgency"],
        "recommended_order_qty": forecast["recommended_order_qty"],
        "restock_deadline": forecast["restock_deadline"],
        "festival_alert": {
            "festival_name": festival_ctx["festival_name"],
            "days_away": festival_ctx["days_to_festival"],
            "demand_multiplier": forecast["demand_multiplier"],
            "affected_products": festival_ctx["affected_products"],
            "in_prep_window": festival_ctx["in_prep_window"],
        },
        "top_pick": ranked[0],
        "alternatives": ranked[1:],
        "feature_importance": risk_model.feature_importance(),
    }


@app.get("/simulate_savings")
async def simulate_savings():
    """60-day Monte Carlo simulation: Traditional vs GraminRoute."""
    std_cash, std_stock = 50000.0, 50
    ai_cash, ai_stock = 50000.0, 50
    base_cost, bulk_cost, delivery_fee, pooled_fee, margin = 80.0, 75.0, 100.0, 25.0, 20.0
    history = []

    for day in range(1, 61):
        is_festival = 20 <= day <= 25
        daily_demand = random.randint(2, 8) + (10 if is_festival else 0)

        # Traditional (reactive)
        if std_stock < 20:
            std_cash -= 40 * base_cost + delivery_fee
            std_stock += 40

        # GraminRoute (predictive + pooling)
        target = 60 if is_festival else 15
        if ai_stock < target:
            needed = target - ai_stock
            pooled = needed >= 40 or random.random() > 0.3
            ai_cash -= (needed * bulk_cost + pooled_fee) if pooled else (needed * base_cost + delivery_fee)
            ai_stock += needed

        # Sales
        std_sold = min(std_stock, daily_demand)
        ai_sold = min(ai_stock, daily_demand)
        std_stock -= std_sold
        std_cash += std_sold * (base_cost + margin)
        ai_stock -= ai_sold
        ai_cash += ai_sold * (base_cost + margin)

        history.append({
            "day": f"Day {day}",
            "Traditional": round(std_cash + std_stock * base_cost),
            "GraminRoute": round(ai_cash + ai_stock * base_cost),
            "isFestival": is_festival,
        })

    return history


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ml"}


# ── Background Kafka Consumer ───────────────

async def consume_inventory_events():
    """Consume inventory.updated events and update demand stats in Redis."""
    await asyncio.sleep(5)  # Wait for Kafka to be ready

    try:
        consumer = AIOKafkaConsumer(
            "inventory.updated",
            bootstrap_servers=KAFKA_SERVERS,
            group_id="ml_analytics",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
        )
        await consumer.start()
        log.info("✅ ML Kafka consumer started (inventory.updated)")

        async for msg in consumer:
            try:
                event = msg.value
                shop_id = event["shop_id"]
                sku_id = event["sku_id"]

                # Update exponentially weighted average demand in Redis
                key = f"demand_ewa:{shop_id}:{sku_id}"
                current = await redis_conn.get(key)
                daily_sales = event.get("daily_sales", 5)

                if current:
                    # EWA: new = alpha * observed + (1-alpha) * old
                    alpha = 0.3
                    new_ewa = alpha * daily_sales + (1 - alpha) * float(current)
                else:
                    new_ewa = float(daily_sales)

                await redis_conn.set(key, str(round(new_ewa, 2)), ex=86400 * 7)
                log.debug(f"📊 EWA updated: {shop_id}/{sku_id} = {new_ewa:.2f}")

            except Exception as e:
                log.error(f"❌ ML consumer error: {e}")

    except Exception as e:
        log.error(f"❌ ML Kafka consumer failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
