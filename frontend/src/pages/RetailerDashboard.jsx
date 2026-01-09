import React, { useState, useEffect } from 'react';
import { ShoppingCart, Package, TrendingUp, LogOut, Upload, CheckCircle, Clock, X, Check } from 'lucide-react';
import { fetchInventory, fetchOrders, createOrder } from '../services/api';
import { CATALOG } from '../utils/helpers';
import ProductCard from './ProductCard';
import SimulationChart from '../components/SimulationChart';

const RetailerDashboard = ({ user, onLogout }) => {
  const [activeTab, setActiveTab] = useState('home');
  const [stockData, setStockData] = useState([]);
  const [orders, setOrders] = useState([]);
  const [aiRecommendations, setAiRecommendations] = useState([]);
  const [cart, setCart] = useState([]);
  const [showCatalog, setShowCatalog] = useState(false);

  // Load Data
  useEffect(() => {
    const loadData = async () => {
      const stock = await fetchInventory(user.id);
      const orderList = await fetchOrders(user.id);
      
      setStockData(stock);
      // Sort orders by date descending (simple client-side sort)
      setOrders(orderList.sort((a,b) => new Date(b.date) - new Date(a.date)));

      // Generate AI Recommendations based on fetched stock
      const lowStock = stock.filter(item => item.current_stock < item.reorder_level);
      setAiRecommendations(lowStock.map(item => ({
        product_id: item.id,
        product_name: item.name,
        current: item.current_stock,
        recommended: Math.max(item.reorder_level + 5, Math.ceil(item.reorder_level * 1.5)),
        urgency: item.current_stock < item.reorder_level * 0.5 ? 'HIGH' : 'MEDIUM',
        unit_price: item.unit_price,
        estimated_cost: Math.max(item.reorder_level + 5, Math.ceil(item.reorder_level * 1.5)) * item.unit_price
      })));
    };
    loadData();
  }, [user.id]);

  const placeOrder = async (items, source) => {
    for (const item of items) {
        const orderData = {
            retailer_id: user.id,
            retailer_name: user.name,
            retailer_location: user.location,
            product_id: item.product_id,
            product_name: item.product_name,
            quantity: item.quantity,
            unit_price: item.unit_price,
            total_amount: item.total || (item.quantity * item.unit_price),
            status: 'pending',
            paid: 0,
            date: new Date().toISOString().split('T')[0],
            source: source
        };
        await createOrder(orderData);
    }
    // Refresh Data
    const newOrders = await fetchOrders(user.id);
    setOrders(newOrders.sort((a,b) => new Date(b.date) - new Date(a.date)));
    alert("Order Placed Successfully!");
  };

  const handleAcceptAIOrder = async (rec) => {
    await placeOrder([{
        product_id: rec.product_id,
        product_name: rec.product_name,
        quantity: rec.recommended,
        unit_price: rec.unit_price,
    }], 'AI Suggested');
    setAiRecommendations(prev => prev.filter(r => r.product_id !== rec.product_id));
  };

  const handlePlaceCartOrder = async () => {
    await placeOrder(cart, 'Manual Order');
    setCart([]);
    setShowCatalog(false);
    setActiveTab('orders');
  };

  // ... (Rendering logic similar to original, keeping it concise) ...
  // Keeping the layout structure but utilizing the new data sources
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow px-4 py-4 flex justify-between items-center">
        <div>
           <h1 className="text-2xl font-bold text-green-700">{user.name}</h1>
           <p className="text-sm text-gray-600">{user.location}</p>
        </div>
        <div className="flex gap-4">
            <button onClick={() => setShowCatalog(true)} className="bg-green-600 text-white px-4 py-2 rounded flex items-center gap-2">
                <ShoppingCart size={20} /> Catalog {cart.length > 0 && `(${cart.length})`}
            </button>
            <button onClick={onLogout} className="text-gray-600"><LogOut /></button>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white border-t px-4 flex gap-6">
          {['home', 'stock', 'orders'].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} 
                className={`py-4 px-2 border-b-2 font-semibold capitalize ${activeTab === tab ? 'border-green-600 text-green-600' : 'border-transparent'}`}>
                {tab}
              </button>
          ))}
      </div>

      <div className="max-w-7xl mx-auto p-4 py-8">
        {activeTab === 'home' && (
            <div className="space-y-6">
                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-white p-6 rounded shadow flex justify-between">
                        <div><p className="text-gray-600">Total Stock Value</p><p className="text-3xl font-bold">₹{stockData.reduce((s, i) => s + (i.current_stock * i.unit_price), 0).toLocaleString()}</p></div>
                        <Package className="text-green-600" size={40} />
                    </div>
                    {/* Add other stats here... */}
                </div>

                {/* AI Recommendations */}
                {aiRecommendations.length > 0 ? (
                    <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl shadow-lg p-6 text-white">
                        <div className="flex items-center gap-3 mb-4"><TrendingUp /><h2 className="text-2xl font-bold">AI Stock Recommendations</h2></div>
                        <div className="space-y-3">
                            {aiRecommendations.map((rec, idx) => (
                                <div key={idx} className="bg-white/10 backdrop-blur rounded p-4 flex justify-between items-center">
                                    <div>
                                        <p className="font-bold">{rec.product_name}</p>
                                        <p className="text-sm">Rec: {rec.recommended} units | Cost: ₹{rec.estimated_cost}</p>
                                    </div>
                                    <div className="flex gap-2">
                                        <button onClick={() => handleAcceptAIOrder(rec)} className="bg-green-500 p-2 rounded"><Check size={16}/></button>
                                        <button onClick={() => setAiRecommendations(aiRecommendations.filter(r => r.product_id !== rec.product_id))} className="bg-red-500 p-2 rounded"><X size={16}/></button>
                                    </div>
                                </div>
                            ))}
                        </div>
                        <SimulationChart />
                    </div>
                    
                ) : <div className="p-6 bg-green-50 rounded text-center text-green-800"><CheckCircle className="inline mb-2"/> All stock healthy!</div>}
            </div>
        )}

        {activeTab === 'stock' && (
            <div className="bg-white rounded shadow p-6 overflow-x-auto">
                <table className="w-full text-left">
                    <thead className="bg-gray-50"><tr><th className="p-3">Product</th><th className="p-3">Stock</th><th className="p-3">Status</th></tr></thead>
                    <tbody>
                        {stockData.map(item => (
                            <tr key={item.docId} className="border-t">
                                <td className="p-3">{item.name}</td>
                                <td className="p-3 font-bold">{item.current_stock}</td>
                                <td className="p-3">{item.current_stock < item.reorder_level ? <span className="text-red-500 font-bold">Low</span> : <span className="text-green-500">OK</span>}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        )}

        {activeTab === 'orders' && (
            <div className="space-y-4">
                {orders.map(order => (
                    <div key={order.docId} className="bg-white p-4 rounded shadow flex justify-between">
                        <div>
                            <p className="font-bold">{order.product_name}</p>
                            <p className="text-xs text-gray-500">{order.docId}</p>
                        </div>
                        <div className="text-right">
                            <p className="font-bold">₹{order.total_amount}</p>
                            <span className={`text-xs px-2 py-1 rounded ${order.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100'}`}>{order.status}</span>
                        </div>
                    </div>
                ))}
            </div>
        )}
      </div>

      {/* Catalog Modal */}
      {showCatalog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white w-full max-w-4xl h-[80vh] flex flex-col rounded-xl">
                <div className="p-4 border-b flex justify-between"><h2 className="text-xl font-bold">Catalog</h2><button onClick={() => setShowCatalog(false)}><X/></button></div>
                <div className="flex-1 overflow-y-auto p-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                    {CATALOG.map(p => (
                        <ProductCard key={p.id} product={p} onAddToCart={(prod, qty) => {
                            setCart(prev => [...prev, { product_id: prod.id, product_name: prod.name, quantity: qty, unit_price: prod.unit_price, total: prod.unit_price * qty }]);
                            alert("Added to cart");
                        }} />
                    ))}
                </div>
                {cart.length > 0 && (
                    <div className="p-4 border-t bg-gray-50 flex justify-between items-center">
                        <p className="font-bold">Total: ₹{cart.reduce((a,b)=>a+b.total,0)}</p>
                        <button onClick={handlePlaceCartOrder} className="bg-green-600 text-white px-6 py-2 rounded">Place Order</button>
                    </div>
                )}
            </div>
        </div>
      )}
    </div>
  );
};

export default RetailerDashboard;