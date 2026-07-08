"""
GraminRoute — Notify Service
==============================
Consumes: pool.formed + dispatch.sent (Kafka)

- pool.formed → push "join this pool" offer to each shop via WebSocket
- dispatch.sent → push "on the way" confirmation to shops
- Falls back to SMS (mock/logger) for shops with no active WebSocket

This service makes Ramesh's phone buzz.
"""

import os
import json
import logging
import asyncio

from aiokafka import AIOKafkaConsumer
import httpx

# ── Config ───────────────────────────────────
KAFKA_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
WS_URL = os.environ.get("WEBSOCKET_SERVICE_URL", "http://localhost:8002")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gramin:gramin@localhost:5432/graminroute")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("notify")


# ── SMS Fallback (Mock) ─────────────────────

def send_sms(shop_id: str, message: str):
    """
    Mock SMS sender. In production, replace with MSG91/Twilio.
    """
    log.info(f"📱 [SMS MOCK] To: {shop_id} | Message: {message}")


# ── Notification Logic ──────────────────────

async def push_to_shop(shop_id: str, event: str, data: dict):
    """Push via WebSocket, fall back to SMS."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{WS_URL}/internal/emit/shop/{shop_id}",
                json={"event": event, "data": data},
                timeout=5.0,
            )
            if resp.status_code == 200:
                log.info(f"  ✅ WS push to shop:{shop_id} — {event}")
                return True
        except Exception as e:
            log.warning(f"  ⚠️ WS failed for {shop_id}: {e}")

    # Fallback: SMS
    send_sms(shop_id, f"[{event}] {json.dumps(data)[:100]}")
    return False


async def push_to_hub(district: str, event: str, data: dict):
    """Push to hub dashboard via WebSocket."""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{WS_URL}/internal/emit/hub/{district}",
                json={"event": event, "data": data},
                timeout=5.0,
            )
            log.info(f"  ✅ WS push to hub:{district} — {event}")
        except Exception as e:
            log.warning(f"  ⚠️ Hub push failed: {e}")


# ── Kafka Event Handlers ────────────────────

async def handle_pool_formed(event: dict):
    """Push pool offer to all shops in the pool."""
    pool_id = event["pool_id"]
    shops = event.get("shops", [])
    district = event.get("district", "Jangaon")

    log.info(f"🔔 Notify: pool.formed — {pool_id} ({len(shops)} shops)")

    for shop_id in shops:
        await push_to_shop(shop_id, "pool_offer", {
            "pool_id": pool_id,
            "sku_name": event.get("sku_name", ""),
            "shops_in_pool": len(shops),
            "total_qty": event.get("total_qty", 0),
            "discount": "15% WHOLESALE" if event.get("total_qty", 0) > 50 else "10% GROUP",
            "message": f"Join pool {pool_id} — {len(shops)} shops near you are ordering {event.get('sku_name', 'this item')}!",
        })

    await push_to_hub(district, "pool_formed", event)


async def handle_dispatch_sent(event: dict):
    """Push delivery confirmation to all shops."""
    pool_id = event.get("pool_id", "")
    shops = event.get("shops", [])
    eta = event.get("eta_minutes", 240)
    district = event.get("district", "Jangaon")

    log.info(f"🚚 Notify: dispatch.sent — {pool_id} ({len(shops)} shops)")

    for shop_id in shops:
        await push_to_shop(shop_id, "dispatch_sent", {
            "pool_id": pool_id,
            "message": f"Your order is on the way! ETA: {eta // 60}h {eta % 60}m",
            "eta_minutes": eta,
        })

    await push_to_hub(district, "dispatch_sent", event)


# ── Main Consumer Loop ──────────────────────

async def main():
    log.info("🚀 Notify Service starting...")

    consumer = AIOKafkaConsumer(
        "pool.formed",
        "dispatch.sent",
        bootstrap_servers=KAFKA_SERVERS,
        group_id="notify_service",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )
    await consumer.start()
    log.info("✅ Kafka consumer started (pool.formed, dispatch.sent)")

    try:
        async for msg in consumer:
            try:
                if msg.topic == "pool.formed":
                    await handle_pool_formed(msg.value)
                elif msg.topic == "dispatch.sent":
                    await handle_dispatch_sent(msg.value)
            except Exception as e:
                log.error(f"❌ Error: {e}", exc_info=True)
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(main())
