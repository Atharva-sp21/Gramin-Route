import React, { useState, useEffect } from 'react';
import { LogOut, Truck, Layers, MapPin, Eye, X, Activity } from 'lucide-react';
import LogisticsMap from '../components/LogisticsMap';
import { fetchOrders, fetchAllRetailers, updateOrderStatus, getOptimizedPools } from '../services/api';

const DistributorDashboard = ({ user, onLogout }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [allOrders, setAllOrders] = useState([]);
  const [orderPools, setOrderPools] = useState([]);
  const [retailers, setRetailers] = useState([]); // Store retailers for Map
  const [selectedPool, setSelectedPool] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        // 1. Fetch All Data Needed
        const [orders, retailerList] = await Promise.all([
          fetchOrders(),
          fetchAllRetailers()
        ]);

        setAllOrders(orders);
        setRetailers(retailerList); // Save retailers state for the map

        // 2. Filter Pending Orders
        const pendingOrders = orders.filter(o => o.status === 'pending');

        // 3. Get Intelligent Pools from Python
        if (pendingOrders.length > 0) {
          const serverPools = await getOptimizedPools(pendingOrders);

          // 4. HYDRATE POOLS
          const hydratedPools = serverPools.map(p => {
            // Find all orders in this pool
            const poolOrders = pendingOrders.filter(o => p.shops.includes(o.retailer_id));
            // Find all retailers in this pool
            const poolRetailers = retailerList.filter(r => p.shops.includes(r.id));
            // Calculate total value
            const poolValue = poolOrders.reduce((sum, o) => sum + (o.total_amount || 0), 0);

            return {
              pool_id: p.pool_id,
              discount: p.discount,
              radius_km: p.radius_km || 0,
              final_amount: poolValue,
              orders: poolOrders,
              retailers: poolRetailers
            };
          });

          setOrderPools(hydratedPools);
        }
      } catch (error) {
        console.error("Error loading dashboard:", error);
      }
      setLoading(false);
    };
    loadData();
  }, []);

  const handleDispatchPool = async (pool) => {
    if (!pool || !pool.orders) return;

    // Update all orders in Firestore
    for (const order of pool.orders) {
      if (order.docId) await updateOrderStatus(order.docId, 'in_transit');
    }

    // Refresh local state instantly
    const updatedOrders = allOrders.map(o =>
      pool.orders.some(po => po.id === o.id) ? { ...o, status: 'in_transit' } : o
    );
    setAllOrders(updatedOrders);
    setOrderPools(orderPools.filter(p => p.pool_id !== pool.pool_id));
    setSelectedPool(null);
    alert(`✅ Pool ${pool.pool_id} dispatched! ${pool.orders.length} orders on the way.`);
  };

  if (loading) return <div className="p-10 text-center flex flex-col items-center justify-center min-h-[50vh]"><Activity className="animate-spin text-blue-600 mb-4" size={40} />Loading Logistics Data...</div>;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-blue-600 text-white p-4 flex justify-between items-center shadow">
        <div><h1 className="text-2xl font-bold">{user.name}</h1><p className="text-blue-100 text-sm">Distributor Dashboard</p></div>
        <button onClick={onLogout} className="flex items-center gap-2 hover:bg-blue-700 px-3 py-1 rounded transition"><LogOut size={18} /> Logout</button>
      </div>

      {/* Navigation */}
      <div className="bg-white border-b px-4 flex gap-6 sticky top-0 z-10 shadow-sm">
        {['overview', 'delivery-pools', 'orders'].map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} className={`py-4 px-2 border-b-2 font-semibold capitalize transition ${activeTab === tab ? 'border-blue-600 text-blue-600' : 'border-transparent hover:text-gray-600'}`}>
            {tab.replace('-', ' ')}
            {tab === 'delivery-pools' && orderPools.length > 0 && <span className="ml-2 bg-orange-500 text-white rounded-full px-2 py-0.5 text-xs">{orderPools.length}</span>}
          </button>
        ))}
      </div>

      <div className="max-w-7xl mx-auto p-4 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <p className="text-gray-500 text-sm font-medium uppercase">Pending Orders</p>
                <p className="text-4xl font-bold text-orange-600 mt-2">{allOrders.filter(o => o.status === 'pending').length}</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <p className="text-gray-500 text-sm font-medium uppercase">Optimized Pools</p>
                <p className="text-4xl font-bold text-purple-600 mt-2">{orderPools.length}</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <p className="text-gray-500 text-sm font-medium uppercase">Revenue (Completed)</p>
                <p className="text-4xl font-bold text-green-600 mt-2">₹{allOrders.filter(o => o.status === 'completed').reduce((a, b) => a + b.total_amount, 0).toLocaleString()}</p>
              </div>
            </div>

            {/* LOGISTICS MAP INTEGRATION */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <MapPin className="text-blue-600" /> Live Logistics Network
              </h2>
              {/* This map now visualizes:
                           1. All 100 Retailers (Red = Critical Stock, Cyan = Healthy)
                           2. Active Pools (Green Lines = Wholesale, Orange = Standard)
                           3. The Hub (Truck Icon)
                        */}
              <LogisticsMap retailers={retailers} pools={orderPools} />

              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
                <div className="p-3 bg-blue-50 rounded-lg border border-blue-100">
                  <strong className="text-blue-800 block mb-1">Graph Insight</strong>
                  Visualizes real-time connections between retailers based on pending orders.
                </div>
                <div className="p-3 bg-green-50 rounded-lg border border-green-100">
                  <strong className="text-green-800 block mb-1">Optimization</strong>
                  Green lines indicate high-efficiency wholesale pools (&gt;50 units).
                </div>
                <div className="p-3 bg-red-50 rounded-lg border border-red-100">
                  <strong className="text-red-800 block mb-1">Risk Alert</strong>
                  Red nodes indicate shops with critical stock levels or high credit risk.
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'delivery-pools' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {orderPools.length === 0 && <p className="text-center text-gray-500 col-span-3 py-10">No pending pools. Great job!</p>}
            {orderPools.map(pool => (
              <div key={pool.pool_id} className="bg-white border border-gray-200 p-5 rounded-xl shadow-sm hover:shadow-md transition group">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-gray-800">{pool.pool_id}</h3>
                    <p className="text-xs text-gray-500">{pool.retailers.length} Shops • {pool.orders.length} Orders</p>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded font-bold ${pool.discount.includes('WHOLESALE') ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}>
                    {pool.discount}
                  </span>
                </div>

                <div className="space-y-2 text-sm text-gray-600 mb-5 bg-gray-50 p-3 rounded-lg">
                  <div className="flex justify-between"><span>Max Radius:</span> <span className="font-medium text-gray-900">{pool.radius_km.toFixed(2)} km</span></div>
                  <div className="flex justify-between"><span>Total Value:</span> <span className="font-bold text-gray-900">₹{pool.final_amount.toLocaleString()}</span></div>
                </div>

                <div className="flex gap-3">
                  <button onClick={() => setSelectedPool(pool)} className="flex-1 bg-white border border-blue-200 text-blue-600 py-2 rounded-lg hover:bg-blue-50 font-semibold flex items-center justify-center gap-2 text-sm transition">
                    <Eye size={16} /> Details
                  </button>
                  <button onClick={() => handleDispatchPool(pool)} className="flex-1 bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 font-semibold flex items-center justify-center gap-2 text-sm transition shadow-sm">
                    <Truck size={16} /> Dispatch
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'orders' && (
          <div className="space-y-3">
            {allOrders.map(order => (
              <div key={order.docId} className="bg-white p-4 rounded-lg border border-gray-100 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <p className="font-bold text-gray-800">{order.product_name}</p>
                  <p className="text-sm text-gray-500">{order.retailer_name}</p>
                  <p className="text-xs text-gray-400 mt-1">{order.retailer_location}</p>
                </div>
                <div className="text-right flex items-center gap-4">
                  <div>
                    <p className="font-bold text-gray-800">₹{order.total_amount?.toLocaleString()}</p>
                    <p className="text-xs text-gray-500">{order.quantity} units</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold capitalize ${order.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                      order.status === 'in_transit' ? 'bg-blue-100 text-blue-700' :
                        'bg-green-100 text-green-700'
                    }`}>{order.status.replace('_', ' ')}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pool Details Modal */}
      {selectedPool && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-in fade-in">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl">
            <div className="p-6 border-b flex justify-between items-center bg-gray-50 rounded-t-xl">
              <div>
                <h2 className="text-xl font-bold text-gray-800">Pool Details ({selectedPool.pool_id})</h2>
                <p className="text-sm text-gray-500">Orders grouped by proximity</p>
              </div>
              <button onClick={() => setSelectedPool(null)} className="text-gray-400 hover:text-gray-600 p-1 rounded-full hover:bg-gray-200 transition"><X size={24} /></button>
            </div>

            <div className="p-6 overflow-y-auto space-y-4">
              {/* Map through Retailers */}
              {selectedPool.retailers && selectedPool.retailers.length > 0 ? (
                selectedPool.retailers.map((r, i) => (
                  <div key={r.id || i} className="flex items-start gap-4 border p-4 rounded-lg hover:bg-gray-50 transition">
                    <div className="bg-red-100 text-red-600 p-2 rounded-full">
                      <MapPin size={20} />
                    </div>
                    <div className="flex-1">
                      <p className="font-bold text-gray-800">{r.name}</p>
                      <p className="text-sm text-gray-600">{r.location}</p>
                      <p className="text-xs text-gray-400 mt-1">ID: {r.id}</p>
                    </div>
                    <div className="text-right">
                      {(() => {
                        const myOrder = selectedPool.orders.find(o => o.retailer_id === r.id);
                        return myOrder ? (
                          <>
                            <p className="font-bold text-gray-800">₹{myOrder.total_amount?.toLocaleString()}</p>
                            <p className="text-xs text-blue-600 font-medium">{myOrder.product_name}</p>
                          </>
                        ) : <span className="text-xs text-gray-400">No active order</span>
                      })()}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-10 text-gray-500">Retailer details not found. Check console.</div>
              )}
            </div>

            <div className="p-6 border-t bg-gray-50 rounded-b-xl">
              <button onClick={() => handleDispatchPool(selectedPool)} className="w-full bg-green-600 text-white py-3 rounded-lg font-bold hover:bg-green-700 transition shadow-lg shadow-green-200 flex items-center justify-center gap-2">
                <Truck size={20} /> Confirm Dispatch Route
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DistributorDashboard;