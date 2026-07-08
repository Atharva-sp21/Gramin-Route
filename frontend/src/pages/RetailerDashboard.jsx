import React, { useState, useEffect, useRef } from 'react';
import {
  ShoppingCart, Package, TrendingUp, LogOut, Truck,
  Home, History, Zap, Users, X, Check
} from 'lucide-react';
import { fetchInventory, fetchOrders, createOrder, getAIRecommendation } from '../services/api';
import SimulationChart from '../components/SimulationChart';
import { io } from 'socket.io-client';

const CATALOG = [
  { id: 'P001', name: 'Rice (50kg)',        unit_price: 2500 },
  { id: 'P002', name: 'Wheat Flour (40kg)', unit_price: 1800 },
  { id: 'P003', name: 'Sugar (50kg)',        unit_price: 2200 },
  { id: 'P004', name: 'Cooking Oil (15L)',  unit_price: 1500 },
  { id: 'P005', name: 'Pulses Mix (25kg)',  unit_price: 3000 },
];

const RetailerDashboard = ({ user, onLogout }) => {
  const [activeTab, setActiveTab]   = useState('home');
  const [stockData, setStockData]   = useState([]);
  const [orders, setOrders]         = useState([]);
  const [aiInsight, setAiInsight]   = useState(null);
  const [pooledOffer, setPooledOffer] = useState(null);
  const [cartQty, setCartQty]       = useState({});
  const [loading, setLoading]       = useState(false);
  const socketRef = useRef(null);

  // ── WebSocket: join shop room, listen for real-time pool offers ──────────────
  useEffect(() => {
    const socket = io("http://localhost:8002", { transports: ["websocket"] });
    socketRef.current = socket;

    socket.on("connect", () => {
      socket.emit("join_room", { room: `shop:${user.id}` });
      console.log("[ws] joined room shop:" + user.id);
    });

    // Real-time pool offer from pool formation service
    socket.on("pool_offer", (data) => {
      console.log("[ws] pool_offer:", data);
      setPooledOffer({
        poolId:         data.pool_id,
        item:           { name: data.sku_name, id: data.sku_id },
        neighbors:      (data.shops_in_pool ?? 1) - 1,
        standard_price: Math.round((data.total_qty > 0 ? 2500 : 2500)),
        pooled_price:   Math.round(2500 * 0.85),
        savings_per_unit: Math.round(2500 * 0.15),
        discount:       data.discount,
        expires_in:     data.expires_in_seconds,
      });
    });

    // Real-time dispatch confirmation
    socket.on("dispatch_sent", (data) => {
      alert(`🚚 ${data.message}`);
    });

    return () => socket.disconnect();
  }, [user.id]);

  // ── Load inventory + AI recommendation on mount ──────────────────────────────
  const loadData = async () => {
    const [stock, orderList] = await Promise.all([
      fetchInventory(user.id),
      fetchOrders(user.id),
    ]);
    setStockData(stock);
    setOrders(orderList);

    // Find most critical (lowest stock) item
    if (!aiInsight && stock.length > 0) {
      const critical = stock.slice().sort((a, b) => a.current_stock - b.current_stock)[0];
      if (critical) {
        const aiResponse = await getAIRecommendation({
          id:            user.id,
          lat:           user.lat,
          lon:           user.lon,
          current_stock: critical.current_stock,
          daily_sales:   critical.daily_sales ?? 5,
          lead_time_days: 3,
          product_name:  critical.name,
        });

        if (aiResponse) {
          setAiInsight({ ...aiResponse, item: critical });
          // Build pool offer from ML recommendation
          const sp = critical.unit_price ?? 2500;
          const pp = Math.round(sp * 0.85);
          setPooledOffer({
            poolId:          null, // will be filled when WS arrives
            item:            critical,
            neighbors:       3,
            standard_price:  sp,
            pooled_price:    pp,
            savings_per_unit: sp - pp,
            discount:        "15% WHOLESALE",
          });
        }
      }
    }
  };

  useEffect(() => { loadData(); }, [user]);

  // ── Accept pool ───────────────────────────────────────────────────────────────
  const handleAcceptPool = async () => {
    if (!pooledOffer) return;
    setLoading(true);
    const qty = aiInsight?.recommended_order_qty ?? 20;
    try {
      if (pooledOffer.poolId) {
        // Join the real pool via gateway
        const { joinPool } = await import('../services/api');
        await joinPool(pooledOffer.poolId, qty);
      } else {
        // Fallback: create a standalone order
        await createOrder({
          retailer_id: user.id,
          product_id:  pooledOffer.item.id ?? 'P001',
          quantity:    qty,
        });
      }
      alert(`🎉 POOL JOINED! You saved ₹${(pooledOffer.savings_per_unit * qty).toLocaleString()}`);
      setAiInsight(null);
      setPooledOffer(null);
      await loadData();
    } catch (e) {
      console.error(e);
      alert("Failed to join pool: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDenyPool = () => {
    if (window.confirm("Reject savings? You will pay full delivery fees later.")) {
      setAiInsight(null);
      setPooledOffer(null);
    }
  };

  const handlePlaceOrder = async (product) => {
    const qty = parseInt(cartQty[product.id] || 0);
    if (qty <= 0) return alert("Enter Qty");
    await createOrder({ retailer_id: user.id, product_id: product.id, quantity: qty });
    alert("Order placed");
    await loadData();
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-800 flex flex-col">

      {/* HEADER */}
      <div className="bg-white shadow px-6 py-4 flex justify-between items-center sticky top-0 z-50">
        <div>
          <h1 className="text-2xl font-extrabold text-green-700 tracking-tight">GraminRoute</h1>
          <p className="text-xs text-gray-500 font-medium">SMART RETAILER NETWORK</p>
        </div>
        <div className="flex bg-gray-100 rounded-lg p-1 gap-1">
          {[['home','Home',Home],['catalog','Catalog',ShoppingCart],['history','Orders',History]].map(([tab,label,Icon]) => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-md text-sm font-semibold flex items-center gap-2 transition ${activeTab === tab ? 'bg-white shadow text-green-700' : 'text-gray-500'}`}>
              <Icon size={16} /> {label}
            </button>
          ))}
        </div>
        <button onClick={onLogout} className="text-gray-400 hover:text-red-500"><LogOut size={16} /></button>
      </div>

      <div className="max-w-6xl mx-auto w-full p-6 space-y-8 flex-grow">

        {/* HOME TAB */}
        {activeTab === 'home' && (
          <>
            {pooledOffer && (
              <div className="bg-white rounded-2xl shadow-xl border border-blue-100 overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white px-6 py-3 flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <Zap size={20} className="text-yellow-300" fill="currentColor" />
                    <span className="font-bold tracking-wide">SMART POOL DETECTED</span>
                  </div>
                  <div className="bg-white/20 px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1">
                    <Users size={12} /> {pooledOffer.neighbors} Neighbors Joining
                  </div>
                </div>

                <div className="p-6 flex flex-col md:flex-row gap-8 items-center">
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">
                      Restock {pooledOffer.item?.name ?? pooledOffer.item?.sku_name}
                    </h2>
                    <p className="text-gray-600 mb-4">
                      Your stock is low ({aiInsight?.days_until_stockout ?? '?'} days left).
                      A truck is passing Jangaon and {pooledOffer.neighbors} other shops are ordering.
                    </p>
                    {aiInsight?.festival_alert?.in_prep_window && (
                      <div className="mb-3 bg-orange-50 border border-orange-200 rounded-lg px-4 py-2 text-sm text-orange-800 font-medium">
                        🎉 {aiInsight.festival_alert.festival_name} in {aiInsight.festival_alert.days_away} days — demand expected to rise {aiInsight.festival_alert.demand_multiplier}×
                      </div>
                    )}
                    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                      <p className="text-xs font-bold text-gray-500 uppercase mb-2">Price Breakdown (Per Unit)</p>
                      <div className="flex items-center gap-6">
                        <div>
                          <p className="text-sm text-gray-400 line-through">Standard</p>
                          <p className="text-lg font-bold text-gray-400">₹{pooledOffer.standard_price}</p>
                        </div>
                        <div className="text-2xl text-gray-300">→</div>
                        <div>
                          <p className="text-sm text-green-600 font-bold">Pool Price</p>
                          <p className="text-2xl font-extrabold text-green-700">₹{pooledOffer.pooled_price}</p>
                        </div>
                        <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs font-bold ml-auto">
                          {pooledOffer.discount ?? 'SAVE 15%'}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="min-w-[280px] flex flex-col gap-3">
                    <div className="text-center mb-2">
                      <span className="text-sm text-gray-500">
                        Proposed Order: {aiInsight?.recommended_order_qty ?? 20} Units
                      </span>
                      <div className="text-3xl font-bold text-blue-900 mt-1">
                        ₹{((aiInsight?.recommended_order_qty ?? 20) * pooledOffer.pooled_price).toLocaleString()}
                      </div>
                      <div className="text-xs text-green-600 font-bold">
                        Total Savings: ₹{((aiInsight?.recommended_order_qty ?? 20) * pooledOffer.savings_per_unit).toLocaleString()}
                      </div>
                    </div>
                    <button onClick={handleAcceptPool} disabled={loading}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-bold shadow-lg flex justify-center items-center gap-2 transition">
                      <Check size={20} /> Accept & Join Pool
                    </button>
                    <button onClick={handleDenyPool}
                      className="w-full bg-white border border-gray-200 text-gray-500 py-3 rounded-lg font-semibold hover:bg-gray-50 flex justify-center items-center gap-2 transition">
                      <X size={20} /> Deny (Pay Full Price)
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <h3 className="font-bold text-lg mb-4 text-gray-700">Profit Projection</h3>
                <SimulationChart />
              </div>
              <div>
                <h3 className="font-bold text-lg mb-4 text-gray-700">Shop Health</h3>
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-6">
                  <p className="text-gray-500 text-xs font-bold uppercase">Total Inventory Value</p>
                  <p className="text-3xl font-bold text-gray-800 mt-2">
                    ₹{stockData.reduce((s, i) => s + (i.current_stock * (i.unit_price ?? 0)), 0).toLocaleString()}
                  </p>
                </div>
                <div className="bg-blue-50 p-6 rounded-xl border border-blue-100">
                  <div className="flex items-start gap-4">
                    <div className="bg-blue-200 p-2 rounded-lg text-blue-700"><Truck size={24} /></div>
                    <div>
                      <h4 className="font-bold text-blue-900">Delivery Status</h4>
                      <p className="text-sm text-blue-800 mt-1">
                        {aiInsight ? `Restock by: ${aiInsight.restock_deadline ?? 'Soon'}` : 'Next truck: Tomorrow 10 AM'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* CATALOG TAB */}
        {activeTab === 'catalog' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {CATALOG.map(product => (
              <div key={product.id} className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <h3 className="font-bold text-lg">{product.name}</h3>
                <p className="text-gray-500 text-sm mb-4">₹{product.unit_price} / unit</p>
                <div className="flex gap-2">
                  <input type="number" placeholder="Qty" className="border p-2 rounded w-20"
                    onChange={e => setCartQty({ ...cartQty, [product.id]: e.target.value })} />
                  <button onClick={() => handlePlaceOrder(product)}
                    className="bg-gray-900 text-white px-4 py-2 rounded flex-1">Order</button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* HISTORY TAB */}
        {activeTab === 'history' && (
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h3 className="font-bold text-lg mb-4">Order History</h3>
            {orders.length === 0 && <p className="text-gray-400 text-sm">No orders yet.</p>}
            {orders.map((o, i) => (
              <div key={o.id ?? i} className="flex justify-between border-b py-3">
                <div>
                  <p className="font-bold">{o.sku_name ?? o.product_name} (x{o.qty ?? o.quantity})</p>
                  <p className="text-xs text-gray-500">{o.created_at?.slice(0, 10) ?? o.date}</p>
                </div>
                <div className="text-right">
                  <p className="font-bold">₹{o.total_amount?.toLocaleString()}</p>
                  {o.source === 'Pool-Deal' && (
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded font-bold">POOLED</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default RetailerDashboard;
