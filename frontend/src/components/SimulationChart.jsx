import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea } from 'recharts';
import { TrendingUp, Zap } from 'lucide-react';

const SimulationChart = () => {
  const [data, setData] = useState([]);
  const [metrics, setMetrics] = useState({ standard: 0, ai: 0, increase: 0 });

  // --- SIMULATION LOGIC (Ported from Python Block 6) ---
  useEffect(() => {
    const runSimulation = () => {
      const days = 60;
      const history = [];
      
      // Setup
      let std_cash = 50000;
      let std_stock = 50;
      let ai_cash = 50000;
      let ai_stock = 50;

      // Prices
      const base_cost = 80;
      const bulk_cost = 75; // Discount!
      const delivery = 100;
      const shared_delivery = 25; // Pooled!
      const margin = 20;

      for (let day = 1; day <= days; day++) {
        // 1. Market Conditions
        const isFestival = day >= 20 && day <= 25;
        let demand = Math.floor(Math.random() * 6) + 2; // 2 to 8
        if (isFestival) demand += 10; // Spike!

        // --- TRADITIONAL LOGIC ---
        if (std_stock < 20) {
            const qty = 40;
            std_cash -= (qty * base_cost) + delivery; // Pays full price
            std_stock += qty;
        }

        // --- GRAMINROUTE AI LOGIC ---
        const target = isFestival ? 50 : 15; // AI predicts demand
        if (ai_stock < target) {
            const needed = target - ai_stock;
            // Pooling Check: High volume or lucky neighbor match
            const isPooled = needed >= 40 || Math.random() > 0.3; 
            
            const cost = isPooled ? bulk_cost : base_cost;
            const del_fee = isPooled ? shared_delivery : delivery;
            
            ai_cash -= (needed * cost) + del_fee;
            ai_stock += needed;
        }

        // --- SALES ---
        // Traditional
        const std_sold = Math.min(std_stock, demand);
        std_stock -= std_sold;
        std_cash += std_sold * (base_cost + margin);

        // AI
        const ai_sold = Math.min(ai_stock, demand);
        ai_stock -= ai_sold;
        ai_cash += ai_sold * (base_cost + margin);

        // Record Net Wealth (Cash + Asset Value)
        history.push({
            day: `Day ${day}`,
            Traditional: Math.round(std_cash + (std_stock * base_cost)),
            GraminRoute: Math.round(ai_cash + (ai_stock * base_cost)),
            isFestival
        });
      }

      setData(history);
      const final = history[history.length - 1];
      setMetrics({
          standard: final.Traditional,
          ai: final.GraminRoute,
          increase: final.GraminRoute - final.Traditional
      });
    };

    runSimulation();
  }, []);

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
      <div className="flex justify-between items-start mb-6">
        <div>
            <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                <TrendingUp className="text-green-600"/> 
                Projected Savings Simulation
            </h2>
            <p className="text-sm text-gray-500">60-Day Comparison: Manual vs. AI Pooling</p>
        </div>
        <div className="text-right">
            <p className="text-sm text-gray-500">Net Profit Increase</p>
            <p className="text-2xl font-bold text-green-600">+₹{metrics.increase.toLocaleString()}</p>
        </div>
      </div>

      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb"/>
            <XAxis dataKey="day" hide />
            <YAxis domain={['auto', 'auto']} tickFormatter={(v) => `₹${v/1000}k`} />
            <Tooltip 
                formatter={(value) => `₹${value.toLocaleString()}`}
                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
            />
            <Legend />
            
            {/* Festival Highlight Zone */}
            <ReferenceArea x1="Day 20" x2="Day 25" fill="#FFEDD5" fillOpacity={0.5} />
            
            <Line 
                type="monotone" 
                dataKey="Traditional" 
                stroke="#9CA3AF" 
                strokeWidth={2} 
                dot={false}
                strokeDasharray="5 5"
            />
            <Line 
                type="monotone" 
                dataKey="GraminRoute" 
                stroke="#16A34A" 
                strokeWidth={3} 
                dot={false} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex gap-4 text-xs text-gray-600 bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-orange-100 border border-orange-200 block"></span>
              <span><strong>Festive Period (Days 20-25):</strong> AI stocks up early, preventing stockouts.</span>
          </div>
          <div className="flex items-center gap-2">
              <Zap size={14} className="text-green-600"/>
              <span><strong>Pooling Effect:</strong> Bulk orders save ~₹75 per delivery.</span>
          </div>
      </div>
    </div>
  );
};

export default SimulationChart;