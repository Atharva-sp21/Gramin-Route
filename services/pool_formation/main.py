"""
GraminRoute — Pool Formation Service
=====================================
Consumes: inventory.updated (Kafka)
Produces: pool.formed (Kafka)
Uses: PostGIS ST_DWithin for 2km geo-clustering
Uses: Redis SADD + EXPIRE 300 for draft pool state

When inventory.updated fires for a high-risk shop:
1. PostGIS: find nearby shops ordering the same SKU within 2km
2. Redis: create draft pool with EXPIRE 300 (auto-cleanup)
3. Postgres: write pool + pool_members records
4. Kafka: publish pool.formed
5. WS: push pool_offer to each shop in the pool
"""

import os
import json
import uuid
import logging
import asyncio

import asyncpg
import redis.asyncio as aioredis
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import httpx

# ── Config ───────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gramin:gramin@localhost:5432/graminroute")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
KAFKA_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
WS_URL = os.environ.get("WEBSOCKET_SERVICE_URL", "http://localhost:8002")

POOL_RADIUS_METERS = 2000   # 2km
POOL_EXPIRE_SECONDS = 300   # 5 minutes
MIN_NEIGHBORS = 1           # at least 1 neighbor to form a pool

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pool_formation")


async def handle_inventory_updated(event: dict, db_pool, redis_conn, producer):
    """
    Core pool formation logic.
    Called when a high-risk inventory update arrives.
    """
    shop_id = event["shop_id"]
    sku_id = event["sku_id"]
    sku_name = event.get("sku_name", sku_id)
    lat = event["lat"]
    lon = event["lon"]
    district = event.get("district", "Jangaon")

    log.info(f"🔍 Pool check for {shop_id}/{sku_id} (risk={event['risk_score']})")

    async with db_pool.acquire() as conn:
        # 1. PostGIS — find nearby shops with low stock on the same SKU
        neighbors = await conn.fetch("""
            SELECT s.id AS shop_id, s.lat, s.lon, s.name,
                   i.qty, i.daily_sales
            FROM inventory i
            JOIN shops s ON s.id = i.shop_id
            WHERE i.sku_id = $1
              AND i.qty < i.daily_sales * 3
              AND s.id != $2
              AND s.user_type = 'retailer'
              AND ST_DWithin(
                    s.location::geography,
                    ST_MakePoint($3, $4)::geography,
                    $5
              )
            ORDER BY ST_Distance(
                s.location::geography,
                ST_MakePoint($3, $4)::geography
            )
            LIMIT 10
        """, sku_id, shop_id, lon, lat, POOL_RADIUS_METERS)

    if len(neighbors) < MIN_NEIGHBORS:
        log.info(f"  ❌ No neighbors found within {POOL_RADIUS_METERS}m — no pool")
        return

    # 2. Redis draft pool — SADD + EXPIRE 300
    pool_id = f"POOL-{uuid.uuid4().hex[:8].upper()}"
    redis_key = f"pool:{sku_id}:{district}"

    all_shop_ids = [shop_id] + [n["shop_id"] for n in neighbors]
    await redis_conn.sadd(redis_key, *all_shop_ids)
    await redis_conn.expire(redis_key, POOL_EXPIRE_SECONDS)

    log.info(f"  📦 Redis draft: {redis_key} = {all_shop_ids} (expires in {POOL_EXPIRE_SECONDS}s)")

    # 3. Write pool record to Postgres
    center_lat = sum([lat] + [float(n["lat"]) for n in neighbors]) / (len(neighbors) + 1)
    center_lon = sum([lon] + [float(n["lon"]) for n in neighbors]) / (len(neighbors) + 1)
    total_qty = sum(n["daily_sales"] * 3 for n in neighbors) + (event.get("qty", 10))

    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO pools (id, sku_id, sku_name, district, status, center_lat, center_lon, total_qty)
            VALUES ($1, $2, $3, $4, 'draft', $5, $6, $7)
        """, pool_id, sku_id, sku_name, district, center_lat, center_lon, total_qty)

        # Add pool members
        for sid in all_shop_ids:
            qty = event.get("qty", 10) if sid == shop_id else next(
                (n["daily_sales"] * 3 for n in neighbors if n["shop_id"] == sid), 10
            )
            await conn.execute("""
                INSERT INTO pool_members (pool_id, shop_id, qty, status)
                VALUES ($1, $2, $3, 'pending')
                ON CONFLICT DO NOTHING
            """, pool_id, sid, qty)

    log.info(f"  📝 Pool {pool_id} written to Postgres ({len(all_shop_ids)} shops)")

    # 4. Publish pool.formed to Kafka
    pool_event = {
        "pool_id": pool_id,
        "sku_id": sku_id,
        "sku_name": sku_name,
        "district": district,
        "shops": all_shop_ids,
        "center_lat": round(center_lat, 6),
        "center_lon": round(center_lon, 6),
        "total_qty": total_qty,
    }
    await producer.send("pool.formed", value=pool_event)
    log.info(f"  📡 Kafka: pool.formed → {pool_id}")

    # 5. Push pool_offer to each shop via WS
    async with httpx.AsyncClient() as client:
        for sid in all_shop_ids:
            try:
                await client.post(
                    f"{WS_URL}/internal/emit/shop/{sid}",
                    json={
                        "event": "pool_offer",
                        "data": {
                            "pool_id": pool_id,
                            "sku_name": sku_name,
                            "shops_in_pool": len(all_shop_ids),
                            "total_qty": total_qty,
                            "discount": "15% WHOLESALE" if total_qty > 50 else "10% GROUP",
                            "expires_in_seconds": POOL_EXPIRE_SECONDS,
                            "center_lat": round(center_lat, 6),
                            "center_lon": round(center_lon, 6),
                        }
                    },
                    timeout=5.0,
                )
            except Exception as e:
                log.warning(f"  ⚠️ Failed to push to {sid}: {e}")

        # Also push to hub dashboard
        try:
            await client.post(
                f"{WS_URL}/internal/emit/hub/{district}",
                json={
                    "event": "pool_formed",
                    "data": pool_event,
                },
                timeout=5.0,
            )
        except Exception as e:
            log.warning(f"  ⚠️ Failed to push to hub: {e}")


async def main():
    """Main consumer loop — consumes inventory.updated from Kafka."""
    log.info("🚀 Pool Formation Service starting...")

    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    log.info("✅ Connected to PostgreSQL")

    # Connect to Redis
    redis_conn = aioredis.from_url(REDIS_URL, decode_responses=True)
    await redis_conn.ping()
    log.info("✅ Connected to Redis")

    # Kafka consumer + producer
    consumer = AIOKafkaConsumer(
        "inventory.updated",
        bootstrap_servers=KAFKA_SERVERS,
        group_id="pool_formation",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    await consumer.start()
    await producer.start()
    log.info("✅ Kafka consumer + producer started")

    try:
        async for msg in consumer:
            try:
                await handle_inventory_updated(msg.value, db_pool, redis_conn, producer)
            except Exception as e:
                log.error(f"❌ Error processing event: {e}", exc_info=True)
    finally:
        await consumer.stop()
        await producer.stop()
        await db_pool.close()
        await redis_conn.close()


if __name__ == "__main__":
    asyncio.run(main())
