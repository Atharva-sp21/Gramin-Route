"""
GraminRoute — Inventory Service
================================
Does two things when a stock update comes in:
1. Atomic PostgreSQL update: UPDATE SET qty = qty + delta
2. XGBoost risk score on the updated state
3. If risk > threshold → publish to Kafka inventory.updated

This service owns the question: "is this shop about to have a problem?"
"""

import os
import json
import logging

import numpy as np
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer

from risk import XGBoostRiskModel, build_features
from festival import get_festival_context

# ── Config ───────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gramin:gramin@localhost:5432/graminroute")
KAFKA_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inventory")

# ── App ──────────────────────────────────────
app = FastAPI(title="GraminRoute Inventory Service", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_pool = None
kafka_producer = None
risk_model = None

RISK_THRESHOLD = 0.5  # publish to Kafka if risk exceeds this


class StockUpdate(BaseModel):
    delta: int  # positive = restock, negative = sale/decrement


@app.on_event("startup")
async def startup():
    global db_pool, kafka_producer, risk_model

    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    log.info("✅ Connected to PostgreSQL")

    kafka_producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await kafka_producer.start()
    log.info("✅ Kafka producer started")

    risk_model = XGBoostRiskModel()
    log.info("✅ XGBoost Risk Model loaded")


@app.on_event("shutdown")
async def shutdown():
    if kafka_producer:
        await kafka_producer.stop()
    if db_pool:
        await db_pool.close()


@app.patch("/stock/{shop_id}/{sku_id}")
async def update_stock(shop_id: str, sku_id: str, update: StockUpdate):
    """
    Atomic stock update → risk scoring → conditional Kafka publish.
    This is the trigger for the entire GraminRoute event chain.
    """
    async with db_pool.acquire() as conn:
        # 1. Atomic update — no race conditions
        row = await conn.fetchrow("""
            UPDATE inventory
            SET qty = GREATEST(0, qty + $1), updated_at = NOW()
            WHERE shop_id = $2 AND sku_id = $3
            RETURNING *
        """, update.delta, shop_id, sku_id)

        if not row:
            raise HTTPException(status_code=404, detail=f"No inventory for {shop_id}/{sku_id}")

        # Get shop info for geo + credit
        shop = await conn.fetchrow(
            "SELECT lat, lon, credit_score, district FROM shops WHERE id = $1", shop_id
        )

    # 2. Compute risk score
    festival_ctx = get_festival_context(row["sku_name"])
    features, _ = build_features(
        current_stock=row["qty"],
        daily_sales=row["daily_sales"],
        lead_time_days=row["lead_time"],
        profit_margin=20.0,   # TODO: store per-shop
        shelf_life=row["shelf_life"],
        credit_score=shop["credit_score"],
        product_name=row["sku_name"],
    )
    risk_score = risk_model.predict_risk(features)
    stock_days_left = row["qty"] / max(row["daily_sales"], 1)
    if stock_days_left < row["lead_time"]:
        risk_score = max(risk_score, 0.85)
    status = "CRITICAL" if risk_score > 0.7 else "WARNING" if risk_score > 0.4 else "STABLE"

    log.info(f"📦 {shop_id}/{sku_id}: qty={row['qty']}, risk={risk_score:.3f} [{status}]")

    # 3. Publish to Kafka if risk is significant
    if risk_score > RISK_THRESHOLD:
        event = {
            "shop_id": shop_id,
            "sku_id": sku_id,
            "sku_name": row["sku_name"],
            "qty": row["qty"],
            "daily_sales": row["daily_sales"],
            "risk_score": round(risk_score, 4),
            "status": status,
            "lat": float(shop["lat"]),
            "lon": float(shop["lon"]),
            "district": shop["district"],
            "festival_context": {
                "festival_name": festival_ctx["festival_name"],
                "days_to_festival": festival_ctx["days_to_festival"],
                "spike_factor": festival_ctx["spike_factor"],
            }
        }
        await kafka_producer.send("inventory.updated", value=event)
        log.info(f"🔥 Kafka: inventory.updated for {shop_id}/{sku_id} (risk={risk_score:.3f})")

    return {
        "shop_id": shop_id,
        "sku_id": sku_id,
        "qty": row["qty"],
        "risk_score": round(risk_score, 4),
        "status": status,
        "festival": festival_ctx["festival_name"],
    }


@app.get("/inventory/{shop_id}")
async def get_inventory(shop_id: str):
    """Get all inventory items for a shop."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT i.*, s.lat, s.lon, s.credit_score, s.village
            FROM inventory i
            JOIN shops s ON s.id = i.shop_id
            WHERE i.shop_id = $1
            ORDER BY i.sku_name
        """, shop_id)

    if not rows:
        raise HTTPException(status_code=404, detail=f"No inventory for {shop_id}")

    result = []
    for row in rows:
        festival_ctx = get_festival_context(row["sku_name"])
        features, _ = build_features(
            current_stock=row["qty"],
            daily_sales=row["daily_sales"],
            lead_time_days=row["lead_time"],
            profit_margin=20.0,
            shelf_life=row["shelf_life"],
            credit_score=row["credit_score"],
            product_name=row["sku_name"],
        )
        risk_score = risk_model.predict_risk(features)
        stock_days_left = row["qty"] / max(row["daily_sales"], 1)
        if stock_days_left < row["lead_time"]:
            risk_score = max(risk_score, 0.85)

        result.append({
            "sku_id": row["sku_id"],
            "sku_name": row["sku_name"],
            "qty": row["qty"],
            "daily_sales": row["daily_sales"],
            "unit_price": float(row["unit_price"]),
            "risk_score": round(risk_score, 4),
            "status": "CRITICAL" if risk_score > 0.7 else "WARNING" if risk_score > 0.4 else "STABLE",
            "festival": festival_ctx["festival_name"],
            "days_to_festival": festival_ctx["days_to_festival"],
        })

    return result


@app.get("/health")
async def health():
    return {"status": "ok", "service": "inventory"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
