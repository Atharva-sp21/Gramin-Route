import React, { useState, useEffect, useRef } from 'react';
import { LogOut, Truck, MapPin, Eye, X, Activity } from 'lucide-react';
import LogisticsMap from '../components/LogisticsMap';
import { fetchOrders, fetchAllRetailers, updateOrderStatus, getOptimizedPools } from '../services/api';
import { io } from 'socket.io-client';

const DistributorDashboard = ({ user, onLogout }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [allOrders, setAllOrders] = useState([]);
  const [orderPools, setOrderPools] = useState([]);
  const [retailers, setRetailers] = useState([]);
  const [selectedPool, setSelectedPool] = useState(null);
  const [loading, setLoading] = useState(true);
  const socketRef = useRef(null);

  // ── WebSocket: join hub room, receive live pool events ──────────────────────
  useEffect(() => {
    const socket = io("http://localhost:8002", {
      transports: ["websocket"],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("[ws] connected, joining hub:Jangaon");
      socket.emit("join_room", { room: "hub:Jangaon" });
    });

    // Re-join room on reconnect
    socket.on("reconnect", () => {
      console.log("[ws] reconnected, rejoining hub:Jangaon");
      socket.emit("join_room", { room: "hub:Jangaon" });
    });

    socket.on("pool_formed", (data) => {
      console.log("[ws] pool_formed:", data);
      setOrderPools(prev => {
        if (prev.find(p => p.id === data.pool_id)) return prev;
        return [{
          id: data.pool_id,
          pool_id: data.pool_id,
          sku_name: data.sku_name,
          status: "draft",
          shop_ids: data.shops,
          total_qty: data.total_qty,
          final_amount: data.total_qty * 2500,
          discount: "15% WHOLESALE",
          radius_km: 2,
          retailers: [],
        }, ...prev];
      });
    });

    socket.on("route_ready", (data) => {
      setOrderPools(prev => prev.map(p =>
        p.id === data.pool_id ? { ...p, route: data } : p
      ));
    });

    socket.on("disconnect", (reason) => {
      console.log("[ws] disconnected:", reason);
    });

    return () => socket.disconnect();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [orders, retailerList] = await Promise.all([
        fetchOrders(),
        fetchAllRetailers(),
      ]);

      console.log("orders from API:", orders);  // ← add this
      console.log("retailerList:", retailerList.length);  // ← and this

      const pools = Array.isArray(orders) ? orders : [];
      setAllOrders(pools);
      setRetailers(retailerList);

      const activePools = pools
        .filter(p => p.status === 'draft' || p.status === 'active')
        .map(p => ({
          ...p,
          pool_id: p.id,
          final_amount: p.total_qty * 2500,
          discount: "15% WHOLESALE",
          radius_km: p.radius_km ?? 2,
          retailers: retailerList.filter(r => p.shop_ids?.includes(r.id)),
        }));

      console.log("activePools:", activePools);  // ← and this

      setOrderPools(activePools);
    } catch (e) {
      console.error("Dashboard load error:", e);
    }
    setLoading(false);
  };
  useEffect(() => { loadData(); }, []);

  const handleDispatchPool = async (pool) => {
    if (!pool) return;
    try {
      // FIX: use pool_id (Postgres), not docId (Firebase)
      await updateOrderStatus(pool.id, 'in_transit');
      setOrderPools(prev => prev.filter(p => p.pool_id !== pool.pool_id));
      setSelectedPool(null);
      alert(`✅ Pool ${pool.pool_id} dispatched! Shops have been notified.`);
    } catch (e) {
      console.error("Dispatch error:", e);
      alert("Dispatch failed: " + e.message);
    }
  };

  if (loading) return (
    <div className="p-10 text-center flex flex-col items-center justify-center min-h-[50vh]">
      <Activity className="animate-spin text-blue-600 mb-4" size={40} />
      Loading Logistics Data...
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-blue-600 text-white p-4 flex justify-between items-center shadow">
        <div>
          <h1 className="text-2xl font-bold">{user.name}</h1>
          <p className="text-blue-100 text-sm">Distributor Dashboard</p>
        </div>
        <button onClick={onLogout} className="flex items-center gap-2 hover:bg-blue-700 px-3 py-1 rounded transition">
          <LogOut size={18} /> Logout
        </button>
      </div>

      <div className="bg-white border-b px-4 flex gap-6 sticky top-0 z-10 shadow-sm">
        {['overview', 'delivery-pools', 'orders'].map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`py-4 px-2 border-b-2 font-semibold capitalize transition ${activeTab === tab ? 'border-blue-600 text-blue-600' : 'border-transparent hover:text-gray-600'
              }`}>
            {tab.replace('-', ' ')}
            {tab === 'delivery-pools' && orderPools.length > 0 &&
              <span className="ml-2 bg-orange-500 text-white rounded-full px-2 py-0.5 text-xs">{orderPools.length}</span>
            }
          </button>
        ))}
      </div>

      <div className="max-w-7xl mx-auto p-4 py-8">

        {activeTab === 'overview' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <p className="text-gray-500 text-sm font-medium uppercase">Active Pools</p>
                <p className="text-4xl font-bold text-orange-600 mt-2">{orderPools.length}</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <p className="text-gray-500 text-sm font-medium uppercase">Total Shops</p>
                <p className="text-4xl font-bold text-purple-600 mt-2">{retailers.length}</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <p className="text-gray-500 text-sm font-medium uppercase">Pool Value</p>
                <p className="text-4xl font-bold text-green-600 mt-2">
                  ₹{orderPools.reduce((s, p) => s + (p.final_amount ?? 0), 0).toLocaleString()}
                </p>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <MapPin className="text-blue-600" /> Live Logistics Network
              </h2>
              <LogisticsMap retailers={retailers} pools={orderPools} />
            </div>
          </div>
        )}

        {activeTab === 'delivery-pools' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {orderPools.length === 0 &&
              <p className="text-center text-gray-500 col-span-3 py-10">No pending pools. All clear!</p>
            }
            {orderPools.map(pool => (
              <div key={pool.pool_id} className="bg-white border border-gray-200 p-5 rounded-xl shadow-sm hover:shadow-md transition">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-gray-800">{pool.pool_id}</h3>
                    <p className="text-xs text-gray-500">
                      {pool.retailers?.length ?? pool.shops?.length ?? '?'} Shops
                    </p>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded font-bold ${pool.discount?.includes('WHOLESALE') ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                    }`}>{pool.discount}</span>
                </div>
                <div className="space-y-2 text-sm text-gray-600 mb-5 bg-gray-50 p-3 rounded-lg">
                  <div className="flex justify-between">
                    <span>Radius:</span>
                    <span className="font-medium">{(pool.radius_km ?? 0).toFixed(2)} km</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Value:</span>
                    <span className="font-bold">₹{(pool.final_amount ?? 0).toLocaleString()}</span>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button onClick={() => setSelectedPool(pool)}
                    className="flex-1 bg-white border border-blue-200 text-blue-600 py-2 rounded-lg hover:bg-blue-50 font-semibold flex items-center justify-center gap-2 text-sm">
                    <Eye size={16} /> Details
                  </button>
                  <button onClick={() => handleDispatchPool(pool)}
                    className="flex-1 bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 font-semibold flex items-center justify-center gap-2 text-sm shadow-sm">
                    <Truck size={16} /> Dispatch
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'orders' && (
          <div className="space-y-3">
            {allOrders.length === 0 && <p className="text-gray-400 text-sm">No orders found.</p>}
            {allOrders.map((order, i) => (
              <div key={order.id ?? i} className="bg-white p-4 rounded-lg border border-gray-100 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <p className="font-bold text-gray-800">{order.sku_name ?? order.product_name}</p>
                  <p className="text-sm text-gray-500">{order.shop_id ?? order.retailer_name}</p>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="font-bold">₹{order.total_amount?.toLocaleString()}</p>
                    <p className="text-xs text-gray-500">{order.qty ?? order.quantity} units</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold capitalize ${order.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                    order.status === 'dispatched' ? 'bg-blue-100 text-blue-700' :
                      'bg-green-100 text-green-700'
                    }`}>{order.status}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pool details modal */}
      {selectedPool && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl">
            <div className="p-6 border-b flex justify-between items-center bg-gray-50 rounded-t-xl">
              <div>
                <h2 className="text-xl font-bold text-gray-800">Pool {selectedPool.pool_id}</h2>
                <p className="text-sm text-gray-500">Shops grouped by proximity</p>
              </div>
              <button onClick={() => setSelectedPool(null)} className="text-gray-400 hover:text-gray-600"><X size={24} /></button>
            </div>
            <div className="p-6 overflow-y-auto space-y-4">
              {(selectedPool.retailers?.length > 0 ? selectedPool.retailers : []).map((r, i) => (
                <div key={r.id ?? i} className="flex items-start gap-4 border p-4 rounded-lg hover:bg-gray-50">
                  <div className="bg-red-100 text-red-600 p-2 rounded-full"><MapPin size={20} /></div>
                  <div className="flex-1">
                    <p className="font-bold text-gray-800">{r.name}</p>
                    <p className="text-sm text-gray-600">{r.village ?? r.location}</p>
                    <p className="text-xs text-gray-400 mt-1">ID: {r.id}</p>
                  </div>
                </div>
              ))}
              {(!selectedPool.retailers || selectedPool.retailers.length === 0) && (
                <p className="text-center py-6 text-gray-400">
                  {(selectedPool.shops ?? []).join(', ') || 'No shop details available'}
                </p>
              )}
            </div>
            <div className="p-6 border-t bg-gray-50 rounded-b-xl">
              <button onClick={() => handleDispatchPool(selectedPool)}
                className="w-full bg-green-600 text-white py-3 rounded-lg font-bold hover:bg-green-700 transition flex items-center justify-center gap-2">
                <Truck size={20} /> Confirm Dispatch
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DistributorDashboard;
