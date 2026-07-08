/**
 * GraminRoute API client
 * All calls go to the Python gateway (port 8080).
 * Token is stored in memory — no Firebase dependency.
 */

const GATEWAY = "http://localhost:8080";

// ── Token store ───────────────────────────────────────────────────────────────
let _token = null;

function setToken(t) { _token = t; }
function getToken() { return _token; }

function authHeaders() {
  return {
    "Content-Type": "application/json",
    ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {}),
  };
}

async function apiFetch(path, options = {}) {
  const res = await fetch(`${GATEWAY}${path}`, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers ?? {}) },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Auth ──────────────────────────────────────────────────────────────────────
export const loginUser = async (userId, password, type) => {
  try {
    const data = await apiFetch("/auth/login", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, password }),
    });
    // Store JWT for subsequent calls
    setToken(data.access_token);
    return { ...data.user, type: data.user.user_type };
  } catch (e) {
    console.error("Login failed:", e.message);
    return null;
  }
};

// Stub — seeding is done via `python services/db/seed.py`, not the browser
export const seedDatabase = async () => {
  alert("Run: python services/db/seed.py\n\nThis seeds 100 Jangaon shops into PostgreSQL.\nLogin after: R001 / sharma123 or D001 / dist123");
};

// ── Inventory ─────────────────────────────────────────────────────────────────
export const fetchInventory = async (shopId) => {
  try {
    const rows = await apiFetch(`/inventory/${shopId}`);
    // Normalise field names to match what the dashboard expects
    return rows.map(r => ({
      ...r,
      id: r.sku_id,
      name: r.sku_name,
      current_stock: r.qty,
      unit_price: r.unit_price,
    }));
  } catch (e) {
    console.error("fetchInventory:", e.message);
    return [];
  }
};

// ── Orders ────────────────────────────────────────────────────────────────────
export const fetchOrders = async (shopId = null) => {
  try {
    // Both retailer and distributor use GET /pools — gateway filters by auth
    return await apiFetch("/pools");
  } catch (e) {
    console.error("fetchOrders:", e.message);
    return [];
  }
};

export const createOrder = async (orderData) => {
  // Map frontend payload → gateway stock update endpoint
  return apiFetch(`/stock/${orderData.retailer_id}/${orderData.product_id}`, {
    method: "PATCH",
    body: JSON.stringify({ delta: -orderData.quantity }),
  });
};

export const updateOrderStatus = async (poolId, status) => {
  if (status === "in_transit") {
    return apiFetch(`/pools/${poolId}/dispatch`, { method: "POST" });
  }
};

// ── Pools ─────────────────────────────────────────────────────────────────────
export const joinPool = async (poolId, qty) => {
  return apiFetch(`/pools/${poolId}/join`, {
    method: "POST",
    body: JSON.stringify({ qty }),
  });
};

// Called by DistributorDashboard with pending orders to get ML pools
export const getOptimizedPools = async (orders) => {
  const payload = orders.map(o => ({
    shop_id: o.retailer_id ?? o.shop_id,
    lat: o.retailer_lat ?? o.lat ?? 17.72,
    lon: o.retailer_lon ?? o.lon ?? 79.16,
    qty_needed: o.quantity ?? o.qty_needed ?? 10,
  }));
  try {
    const res = await fetch("http://localhost:8000/pool_orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.ok ? res.json() : [];
  } catch {
    console.error("Pooling engine offline");
    return [];
  }
};

// ── Retailers (for distributor map) ───────────────────────────────────────────
export const fetchAllRetailers = async () => {
  try {
    return await apiFetch("/shops");
  } catch (e) {
    console.error("fetchAllRetailers:", e.message);
    return [];
  }
};

// ── AI / ML ───────────────────────────────────────────────────────────────────
export const getAIRecommendation = async ({ id, shop_id, lat, lon, current_stock, stock, product_name, daily_sales, lead_time_days }) => {
  const payload = {
    shop_id: id ?? shop_id,
    lat: parseFloat(lat),
    lon: parseFloat(lon),
    current_stock: parseInt(current_stock ?? stock ?? 0),
    daily_sales: parseInt(daily_sales ?? 5),
    lead_time_days: parseInt(lead_time_days ?? 3),
    product_name: product_name ?? "Rice (50kg)",
    is_festival: false,
  };
  console.log("📤 AI payload:", payload);
  try {
    const res = await fetch("http://localhost:8004/recommend_distributor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      console.error("AI error:", err);
      return null;
    }
    return res.json();
  } catch (e) {
    console.error("AI Brain offline:", e);
    return null;
  }
};

export const getSimulationData = async () => {
  try {
    const res = await fetch("http://localhost:8004/simulate_savings");
    return res.ok ? res.json() : [];
  } catch { return []; }
};
