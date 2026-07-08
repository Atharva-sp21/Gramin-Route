"""
Seed script — loads jangaon_shops.csv into PostgreSQL + PostGIS.
Also creates the distributor hub (D001) and default product catalog.

Run: python services/db/seed.py
"""

import csv
import sys
import random
import asyncio
from pathlib import Path

import asyncpg
from passlib.hash import bcrypt

import os
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gramin:gramin@127.0.0.1:5432/graminroute")
REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "data" / "jangaon_shops.csv"

CATALOG = [
    {"id": "P001", "name": "Rice (50kg)",        "price": 2500, "shelf_life": 365},
    {"id": "P002", "name": "Wheat Flour (40kg)", "price": 1800, "shelf_life": 180},
    {"id": "P003", "name": "Sugar (50kg)",        "price": 2200, "shelf_life": 365},
    {"id": "P004", "name": "Cooking Oil (15L)",  "price": 1500, "shelf_life": 180},
    {"id": "P005", "name": "Pulses Mix (25kg)",  "price": 3000, "shelf_life": 120},
    {"id": "P006", "name": "Tea Powder (5kg)",   "price": 800,  "shelf_life": 365},
    {"id": "P007", "name": "Spices Mix (10kg)",  "price": 1200, "shelf_life": 180},
    {"id": "P008", "name": "Salt (50kg)",         "price": 500,  "shelf_life": 730},
]

HUB = {
    "id": "D001",
    "name": "GraminRoute Hub",
    "village": "Jangaon",
    "lat": 17.7200,
    "lon": 79.1600,
    "user_type": "distributor",
    "password": "dist123",
}


async def seed():
    print("🌱 GraminRoute — Seeding PostgreSQL...")

    conn = await asyncpg.connect(DATABASE_URL)

    try:
        # Clear existing data
        await conn.execute("DELETE FROM orders")
        await conn.execute("DELETE FROM pool_members")
        await conn.execute("DELETE FROM pools")
        await conn.execute("DELETE FROM inventory")
        await conn.execute("DELETE FROM shops")
        print("  🗑️  Cleared existing data")

        # Insert Hub
        hub_hash = bcrypt.hash(HUB["password"])
        # FIX: ST_MakePoint(lon, lat) — longitude first, latitude second
        await conn.execute("""
            INSERT INTO shops (id, name, village, lat, lon, location, user_type, password_hash)
            VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($5, $4), 4326)::geography, $6, $7)
        """, HUB["id"], HUB["name"], HUB["village"], HUB["lat"], HUB["lon"],
             HUB["user_type"], hub_hash)
        print(f"  🏭 Hub created: {HUB['name']} ({HUB['id']})")

        # Read CSV and insert shops
        if not CSV_PATH.exists():
            print(f"  ❌ CSV not found: {CSV_PATH}")
            sys.exit(1)

        shop_count = 0
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shop_id = f"R{shop_count + 1:03d}"
                village = row.get("Village_Name", "Unknown")
                lat = float(row.get("Latitude", 17.72))
                lon = float(row.get("Longitude", 79.16))
                credit = int(float(row.get("Credit_Score", 700)))
                stock = int(float(row.get("Stock", 50)))

                pwd_hash = bcrypt.hash("sharma123") if shop_id == "R001" else bcrypt.hash("shop123")

                # FIX: ST_MakePoint(lon, lat) — longitude first
                await conn.execute("""
                    INSERT INTO shops (id, name, village, lat, lon, location, credit_score, password_hash)
                    VALUES ($1, $2, $3, $4, $5, ST_SetSRID(ST_MakePoint($5, $4), 4326)::geography, $6, $7)
                    ON CONFLICT (id) DO NOTHING
                """, shop_id, f"Sri Balaji Kirana {shop_count}", village, lat, lon, credit, pwd_hash)

                # Create inventory for each product
                for product in CATALOG:
                    initial_qty = max(1, stock + random.randint(-20, 20))
                    daily_sales = random.randint(2, 10)
                    await conn.execute("""
                        INSERT INTO inventory (shop_id, sku_id, sku_name, qty, daily_sales, unit_price, shelf_life)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (shop_id, sku_id) DO NOTHING
                    """, shop_id, product["id"], product["name"], initial_qty,
                         daily_sales, product["price"], product["shelf_life"])

                shop_count += 1

        print(f"  🏪 {shop_count} shops inserted with inventory")

        shop_n = await conn.fetchval("SELECT COUNT(*) FROM shops")
        inv_n  = await conn.fetchval("SELECT COUNT(*) FROM inventory")
        print(f"\n✅ Seed complete: {shop_n} shops, {inv_n} inventory rows")
        print(f"\n  Login credentials:")
        print(f"  Retailer: R001 / sharma123")
        print(f"  Distributor: D001 / dist123")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())
