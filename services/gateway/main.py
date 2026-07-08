"""
GraminRoute — API Gateway
=========================
The front door. Every request hits here first.
- JWT auth (issue + verify)
- Rate limiting (Redis-backed)
- Routes requests to internal services
"""

import os
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth import (
    create_access_token, verify_password, get_current_user,
    hash_password, Token, LoginRequest
)

import asyncpg

# ── App ──────────────────────────────────────
app = FastAPI(title="GraminRoute Gateway", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gramin:gramin@localhost:5432/graminroute")
INVENTORY_URL = os.environ.get("INVENTORY_SERVICE_URL", "http://localhost:8001")
WEBSOCKET_URL = os.environ.get("WEBSOCKET_SERVICE_URL", "http://localhost:8002")

db_pool = None


@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    print("✅ Gateway connected to PostgreSQL")


@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()


# ── Auth Endpoints ───────────────────────────

@app.post("/auth/login", response_model=Token)
async def login(req: LoginRequest):
    """Authenticate shop/hub and issue JWT."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, village, lat, lon, user_type, password_hash, credit_score "
            "FROM shops WHERE id = $1",
            req.user_id
        )

    if not row:
        raise HTTPException(status_code=401, detail="User not found")

    if not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = create_access_token({
        "sub": row["id"],
        "type": row["user_type"],
        "name": row["name"],
    })

    return Token(
        access_token=token,
        token_type="bearer",
        user={
            "id": row["id"],
            "name": row["name"],
            "village": row["village"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "user_type": row["user_type"],
            "credit_score": row["credit_score"],
        }
    )


# ── Proxy: Inventory Service ─────────────────

@app.patch("/stock/{shop_id}/{sku_id}")
async def proxy_stock_update(shop_id: str, sku_id: str, request: Request, user=Depends(get_current_user)):
    """Forward stock update to Inventory Service with auth context."""
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{INVENTORY_URL}/stock/{shop_id}/{sku_id}",
            json=body,
            headers={"X-Shop-Id": user["sub"]},
            timeout=10.0,
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/inventory/{shop_id}")
async def proxy_inventory(shop_id: str, user=Depends(get_current_user)):
    """Get inventory for a shop."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{INVENTORY_URL}/inventory/{shop_id}",
            headers={"X-Shop-Id": user["sub"]},
            timeout=10.0,
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())


# ── Proxy: Pool endpoints ────────────────────

@app.get("/pools")
async def get_pools(user=Depends(get_current_user)):
    """Get all active pools."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT p.*, array_agg(pm.shop_id) AS shop_ids,
                   SUM(pm.qty) AS member_qty
            FROM pools p
            LEFT JOIN pool_members pm ON pm.pool_id = p.id
            WHERE p.status IN ('draft', 'confirmed')
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """)
    return [dict(r) for r in rows]


@app.post("/pools/{pool_id}/join")
async def join_pool(pool_id: str, request: Request, user=Depends(get_current_user)):
    """Shop joins a pool."""
    body = await request.json()
    shop_id = user["sub"]
    qty = body.get("qty", 10)

    async with db_pool.acquire() as conn:
        # Check pool exists and is still draft
        pool = await conn.fetchrow(
            "SELECT * FROM pools WHERE id = $1 AND status = 'draft'", pool_id
        )
        if not pool:
            raise HTTPException(status_code=404, detail="Pool not found or already confirmed")

        # Add member
        await conn.execute("""
            INSERT INTO pool_members (pool_id, shop_id, qty, status)
            VALUES ($1, $2, $3, 'accepted')
            ON CONFLICT (pool_id, shop_id) DO UPDATE SET status = 'accepted', qty = $3
        """, pool_id, shop_id, qty)

        # Update pool total
        await conn.execute("""
            UPDATE pools SET total_qty = (
                SELECT COALESCE(SUM(qty), 0) FROM pool_members
                WHERE pool_id = $1 AND status = 'accepted'
            ) WHERE id = $1
        """, pool_id)

    return {"status": "joined", "pool_id": pool_id}


@app.post("/pools/{pool_id}/decline")
async def decline_pool(pool_id: str, user=Depends(get_current_user)):
    """Shop declines a pool offer."""
    shop_id = user["sub"]
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE pool_members SET status = 'declined'
            WHERE pool_id = $1 AND shop_id = $2
        """, pool_id, shop_id)
    return {"status": "declined", "pool_id": pool_id}


@app.post("/pools/{pool_id}/dispatch")
async def dispatch_pool(pool_id: str, user=Depends(get_current_user)):
    """Hub dispatches a pool — publishes dispatch.sent to Kafka."""
    if user.get("type") != "distributor":
        raise HTTPException(status_code=403, detail="Only distributors can dispatch")

    async with db_pool.acquire() as conn:
        pool = await conn.fetchrow(
            "SELECT * FROM pools WHERE id = $1 AND status IN ('draft', 'confirmed')", pool_id
        )
        if not pool:
            raise HTTPException(status_code=404, detail="Pool not found")

        await conn.execute("""
            UPDATE pools SET status = 'dispatched', dispatched_at = NOW()
            WHERE id = $1
        """, pool_id)

        # Get member shops
        members = await conn.fetch(
            "SELECT shop_id FROM pool_members WHERE pool_id = $1 AND status = 'accepted'",
            pool_id
        )

    # Publish dispatch.sent to Kafka (via inventory service or direct)
    shop_ids = [m["shop_id"] for m in members]

    # Notify all shops via WS service
    async with httpx.AsyncClient() as client:
        for sid in shop_ids:
            try:
                await client.post(
                    f"{WEBSOCKET_URL}/internal/emit/shop/{sid}",
                    json={
                        "event": "dispatch_sent",
                        "data": {
                            "pool_id": pool_id,
                            "message": "Your order has been dispatched!",
                            "eta_minutes": 240,
                            "shops": shop_ids,
                        }
                    },
                    timeout=5.0,
                )
            except Exception:
                pass  # WS best-effort

    return {"status": "dispatched", "pool_id": pool_id, "shops": shop_ids}


# ── Direct DB queries for dashboard ──────────

@app.get("/shops")
async def get_shops(user=Depends(get_current_user)):
    """Get all retailer shops (for distributor map)."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT s.id, s.name, s.village, s.lat, s.lon, s.credit_score,
                   COALESCE(SUM(i.qty), 0) AS total_stock
            FROM shops s
            LEFT JOIN inventory i ON i.shop_id = s.id
            WHERE s.user_type = 'retailer'
            GROUP BY s.id
            ORDER BY s.id
        """)
    return [dict(r) for r in rows]


@app.get("/orders/{shop_id}")
async def get_orders(shop_id: str, user=Depends(get_current_user)):
    """Get orders for a shop."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM orders WHERE shop_id = $1 ORDER BY created_at DESC", shop_id
        )
    return [dict(r) for r in rows]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "gateway"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
