"""
GraminRoute — Route Optimizer
==============================
Consumes: pool.formed (Kafka)

Takes shop coordinates from PostGIS, runs a nearest-neighbor TSP solver
to find the shortest delivery route: Hub → shops → Hub.

At 4-8 stops this is instant. Writes route_json to the pools table
and pushes the result to the hub dashboard via WebSocket.
"""

import os
import json
import logging
import asyncio
from math import radians, sin, cos, sqrt, atan2

import asyncpg
from aiokafka import AIOKafkaConsumer
import httpx

# ── Config ───────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gramin:gramin@localhost:5432/graminroute")
KAFKA_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
WS_URL = os.environ.get("WEBSOCKET_SERVICE_URL", "http://localhost:8002")

# Hub location (center of Jangaon)
HUB_LAT = 17.7200
HUB_LON = 79.1600

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("route_optimizer")


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def nearest_neighbor_tsp(stops):
    """
    Nearest-neighbor greedy TSP.
    Input: list of (id, lat, lon) tuples.
    Returns: ordered list of stops, total distance.

    Route: Hub → nearest unvisited → ... → Hub
    """
    if not stops:
        return [], 0.0

    unvisited = list(stops)
    route = []
    current = ("HUB", HUB_LAT, HUB_LON)
    total_dist = 0.0

    while unvisited:
        nearest = min(
            unvisited,
            key=lambda s: haversine(current[1], current[2], s[1], s[2])
        )
        dist = haversine(current[1], current[2], nearest[1], nearest[2])
        total_dist += dist
        route.append({
            "shop_id": nearest[0],
            "lat": nearest[1],
            "lon": nearest[2],
            "distance_from_prev_km": round(dist, 2),
        })
        unvisited.remove(nearest)
        current = nearest

    # Return to hub
    return_dist = haversine(current[1], current[2], HUB_LAT, HUB_LON)
    total_dist += return_dist

    return route, round(total_dist, 2)


async def handle_pool_formed(event: dict, db_pool):
    """Process a pool.formed event — compute route and store it."""
    pool_id = event["pool_id"]
    shop_ids = event["shops"]
    district = event.get("district", "Jangaon")

    log.info(f"🗺️  Computing route for {pool_id} ({len(shop_ids)} shops)")

    # Fetch shop coordinates from PostgreSQL
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, lat, lon, name FROM shops
            WHERE id = ANY($1::text[])
        """, shop_ids)

    if not rows:
        log.warning(f"  ❌ No shops found for {pool_id}")
        return

    # Build stops list
    stops = [(row["id"], float(row["lat"]), float(row["lon"])) for row in rows]

    # Run TSP
    route, total_distance = nearest_neighbor_tsp(stops)

    route_json = {
        "pool_id": pool_id,
        "hub": {"lat": HUB_LAT, "lon": HUB_LON, "name": "GraminRoute Hub"},
        "stops": route,
        "total_distance_km": total_distance,
        "estimated_time_minutes": int(total_distance / 30 * 60),  # ~30 km/h avg
    }

    log.info(f"  ✅ Route: Hub → {' → '.join(s['shop_id'] for s in route)} → Hub ({total_distance} km)")

    # Write route to PostgreSQL
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE pools SET route_json = $1::jsonb WHERE id = $2
        """, json.dumps(route_json), pool_id)

    # Push to hub dashboard via WebSocket
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{WS_URL}/internal/emit/hub/{district}",
                json={
                    "event": "route_ready",
                    "data": route_json,
                },
                timeout=5.0,
            )
            log.info(f"  📡 Route pushed to hub:{district}")
        except Exception as e:
            log.warning(f"  ⚠️ Failed to push route to hub: {e}")


async def main():
    log.info("🚀 Route Optimizer starting...")

    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    log.info("✅ Connected to PostgreSQL")

    consumer = AIOKafkaConsumer(
        "pool.formed",
        bootstrap_servers=KAFKA_SERVERS,
        group_id="route_optimizer",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )
    await consumer.start()
    log.info("✅ Kafka consumer started (pool.formed)")

    try:
        async for msg in consumer:
            try:
                await handle_pool_formed(msg.value, db_pool)
            except Exception as e:
                log.error(f"❌ Error: {e}", exc_info=True)
    finally:
        await consumer.stop()
        await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
