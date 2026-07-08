-- =============================================
-- GraminRoute — PostgreSQL + PostGIS Schema
-- =============================================
-- Run order: this file is auto-executed by
-- postgres docker-entrypoint-initdb.d
-- =============================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ── Shops ───────────────────────────────────
-- Each kirana shop with geo coordinates
CREATE TABLE shops (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    village      TEXT,
    district     TEXT DEFAULT 'Jangaon',
    location     GEOGRAPHY(POINT, 4326) NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    credit_score INT DEFAULT 700,
    password_hash TEXT NOT NULL DEFAULT '$2b$12$defaulthashplaceholder',
    user_type    TEXT DEFAULT 'retailer' CHECK (user_type IN ('retailer', 'distributor')),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_shops_location ON shops USING GIST(location);
CREATE INDEX idx_shops_district ON shops(district);

-- ── Inventory ───────────────────────────────
-- Per-shop, per-SKU stock levels
-- qty is updated atomically: UPDATE SET qty = qty - n
CREATE TABLE inventory (
    id          SERIAL PRIMARY KEY,
    shop_id     TEXT NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    sku_id      TEXT NOT NULL,
    sku_name    TEXT NOT NULL,
    qty         INT NOT NULL DEFAULT 0 CHECK (qty >= 0),
    daily_sales INT DEFAULT 5,
    lead_time   INT DEFAULT 3,
    unit_price  NUMERIC(10,2) DEFAULT 0,
    shelf_life  INT DEFAULT 30,
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (shop_id, sku_id)
);

CREATE INDEX idx_inventory_shop ON inventory(shop_id);
CREATE INDEX idx_inventory_sku ON inventory(sku_id);

-- ── Pools ───────────────────────────────────
-- Delivery pools formed by geo-clustering
-- status: draft → confirmed → dispatched → delivered
CREATE TABLE pools (
    id          TEXT PRIMARY KEY,
    sku_id      TEXT NOT NULL,
    sku_name    TEXT,
    district    TEXT NOT NULL DEFAULT 'Jangaon',
    status      TEXT DEFAULT 'draft'
                CHECK (status IN ('draft','confirmed','dispatched','delivered')),
    center_lat  DOUBLE PRECISION,
    center_lon  DOUBLE PRECISION,
    total_qty   INT DEFAULT 0,
    route_json  JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    expires_at  TIMESTAMPTZ DEFAULT NOW() + INTERVAL '5 minutes',
    dispatched_at TIMESTAMPTZ
);

CREATE INDEX idx_pools_status ON pools(status);
CREATE INDEX idx_pools_district ON pools(district);

-- ── Pool Members ────────────────────────────
-- Which shops are in which pool
CREATE TABLE pool_members (
    pool_id   TEXT NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
    shop_id   TEXT NOT NULL REFERENCES shops(id),
    qty       INT NOT NULL DEFAULT 0,
    status    TEXT DEFAULT 'pending'
              CHECK (status IN ('pending','accepted','declined')),
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (pool_id, shop_id)
);

-- ── Orders ──────────────────────────────────
-- Confirmed orders (post-pool or standalone)
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    shop_id     TEXT NOT NULL REFERENCES shops(id),
    pool_id     TEXT REFERENCES pools(id),
    sku_id      TEXT NOT NULL,
    sku_name    TEXT,
    qty         INT NOT NULL,
    unit_price  NUMERIC(10,2),
    total_amount NUMERIC(10,2),
    status      TEXT DEFAULT 'pending'
                CHECK (status IN ('pending','confirmed','in_transit','delivered','cancelled')),
    source      TEXT DEFAULT 'manual',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_orders_shop ON orders(shop_id);
CREATE INDEX idx_orders_status ON orders(status);

-- ── Hub (Distributor) ───────────────────────
-- The distributor hub — stored in shops table with user_type='distributor'
-- This is just a special row in shops, not a separate table
