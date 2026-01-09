// Constants for seeding/defaults
export const CATALOG = [
  { id: 'P001', name: 'Rice (50kg)', category: 'Grains', unit_price: 2500, reorder_level: 10, stock: 'Unlimited', image: '🌾' },
  { id: 'P002', name: 'Wheat Flour (40kg)', category: 'Grains', unit_price: 1800, reorder_level: 15, stock: 'Unlimited', image: '🌾' },
  { id: 'P003', name: 'Sugar (50kg)', category: 'Sweeteners', unit_price: 2200, reorder_level: 12, stock: 'Unlimited', image: '🍬' },
  { id: 'P004', name: 'Cooking Oil (15L)', category: 'Oils', unit_price: 1500, reorder_level: 20, stock: 'Unlimited', image: '🛢️' },
  { id: 'P005', name: 'Pulses Mix (25kg)', category: 'Pulses', unit_price: 3000, reorder_level: 8, stock: 'Unlimited', image: '🫘' },
  { id: 'P006', name: 'Tea Powder (5kg)', category: 'Beverages', unit_price: 800, reorder_level: 25, stock: 'Unlimited', image: '☕' },
  { id: 'P007', name: 'Spices Mix (10kg)', category: 'Spices', unit_price: 1200, reorder_level: 15, stock: 'Unlimited', image: '🌶️' },
  { id: 'P008', name: 'Salt (50kg)', category: 'Condiments', unit_price: 500, reorder_level: 30, stock: 'Unlimited', image: '🧂' },
];

export const calculateDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371.0; // Earth radius in km
  const toRad = (deg) => deg * (Math.PI / 180);
  const dlat = toRad(lat2 - lat1);
  const dlon = toRad(lon2 - lon1);
  const a = Math.sin(dlat / 2) ** 2 + 
            Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dlon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

export const generateOrderPools = (orders, retailers) => {
  const pools = [];
  const processed = new Set();
  let poolCounter = 1;

  orders.forEach((order) => {
    if (processed.has(order.id)) return;

    const retailer = retailers.find(r => r.id === order.retailer_id);
    if (!retailer) return;

    const currentPool = {
      pool_id: `POOL-${String(poolCounter).padStart(3, '0')}`,
      orders: [order],
      retailers: [retailer],
      total_qty: order.quantity,
      total_amount: order.total_amount,
      center_lat: retailer.lat,
      center_lon: retailer.lon,
      radius_km: 0
    };
    processed.add(order.id);

    orders.forEach((neighbor) => {
      if (processed.has(neighbor.id)) return;
      const neighborRetailer = retailers.find(r => r.id === neighbor.retailer_id);
      if (!neighborRetailer) return;

      const dist = calculateDistance(
        retailer.lat, retailer.lon,
        neighborRetailer.lat, neighborRetailer.lon
      );

      if (dist < 3.0) { // 3km clustering
        currentPool.orders.push(neighbor);
        // Avoid duplicate retailer entries in the pool list
        if(!currentPool.retailers.find(r => r.id === neighborRetailer.id)){
            currentPool.retailers.push(neighborRetailer);
        }
        currentPool.total_qty += neighbor.quantity;
        currentPool.total_amount += neighbor.total_amount;
        currentPool.radius_km = Math.max(currentPool.radius_km, dist);
        processed.add(neighbor.id);
      }
    });

    if (currentPool.total_qty > 50) {
      currentPool.discount = "15% WHOLESALE";
      currentPool.discount_amount = Math.round(currentPool.total_amount * 0.15);
      currentPool.final_amount = currentPool.total_amount - currentPool.discount_amount;
    } else {
      currentPool.discount = "STANDARD";
      currentPool.discount_amount = 0;
      currentPool.final_amount = currentPool.total_amount;
    }

    pools.push(currentPool);
    poolCounter++;
  });

  return pools;
};