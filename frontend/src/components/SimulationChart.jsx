import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart
} from 'recharts';
import { TrendingUp, ArrowUpRight } from 'lucide-react';

const SimulationChart = () => {
  // We use useMemo so the simulation runs once and stays stable (doesn't jitter on re-renders)
  const simulationData = useMemo(() => {
    const days = 60;
    const initialCash = 50000;
    const marginPerUnit = 20;
    
    // COSTS
    const baseCost = 80;      // Traditional Price
    const bulkCost = 75;      // GraminRoute Pooled Price
    const deliveryFee = 100;
    const shelfLifeLimit = 10;

    // 1. TRADITIONAL STATE
    let stdCash = initialCash;
    let stdStock = 50;
    let stdBatches = [{ qty: 50, life: shelfLifeLimit }];
    
    // 2. GRAMINROUTE STATE
    let aiCash = initialCash;
    let aiStock = 50;
    let aiBatches = [{ qty: 50, life: shelfLifeLimit }];

    const data = [];

    // HELPER: FIFO SALES LOGIC
    const processDay = (batches, demand) => {
      let sales = 0;
      let remainingDemand = demand;
      
      // Sell oldest stock first
      batches.forEach(batch => {
        if (remainingDemand > 0 && batch.qty > 0) {
          const sold = Math.min(batch.qty, remainingDemand);
          batch.qty -= sold;
          remainingDemand -= sold;
          sales += sold;
        }
      });

      // Rotting Logic (Age decreases)
      const newBatches = batches
        .map(b => ({ ...b, life: b.life - 1 }))
        .filter(b => b.life > 0 && b.qty > 0);

      const totalStock = newBatches.reduce((sum, b) => sum + b.qty, 0);
      return { newBatches, sales, totalStock };
    };

    // --- DAY LOOP ---
    for (let day = 0; day < days; day++) {
      // Demand Simulation
      const isFestival = (day > 20 && day < 25);
      let dailyDemand = Math.floor(Math.random() * (8 - 2) + 2); // Random between 2 and 8
      if (isFestival) dailyDemand += 10;

      // --- TRADITIONAL LOGIC (Reactive) ---
      if (stdStock < 20) {
        const orderQty = 40;
        const cost = (orderQty * baseCost) + deliveryFee; // Full Delivery Fee
        stdCash -= cost;
        stdBatches.push({ qty: orderQty, life: shelfLifeLimit });
        stdStock += orderQty;
      }

      // --- GRAMINROUTE LOGIC (Predictive + Pooling) ---
      const target = isFestival ? 50 : 15; // AI Prediction
      
      if (aiStock < target) {
        let needed = target - aiStock;
        if (!isFestival) needed = Math.min(needed, 10); // Anti-Burn Rule

        // POOLING PROBABILITY (70% chance neighbors exist)
        const neighborParticipation = Math.random() > 0.3;
        
        let unitCost = baseCost;
        let delivery = deliveryFee;

        if (neighborParticipation || needed >= 50) {
          unitCost = bulkCost;       // Cheaper Unit Price
          delivery = deliveryFee / 4; // Split Delivery Fee
        }

        const cost = (needed * unitCost) + delivery;
        aiCash -= cost;
        aiBatches.push({ qty: needed, life: shelfLifeLimit });
        aiStock += needed;
      }

      // PROCESS SALES
      const stdRes = processDay(stdBatches, dailyDemand);
      stdBatches = stdRes.newBatches;
      stdStock = stdRes.totalStock;
      stdCash += (stdRes.sales * (baseCost + marginPerUnit));

      const aiRes = processDay(aiBatches, dailyDemand);
      aiBatches = aiRes.newBatches;
      aiStock = aiRes.totalStock;
      aiCash += (aiRes.sales * (baseCost + marginPerUnit));

      // Record Data for Chart (Net Wealth = Cash + Inventory Value)
      data.push({
        day: `Day ${day + 1}`,
        Traditional: Math.round(stdCash + (stdStock * baseCost)),
        GraminRoute: Math.round(aiCash + (aiStock * baseCost)),
      });
    }

    return data;
  }, []); // Empty dependency array = runs once on mount

  // Calculate Final Stats
  const finalTrad = simulationData[simulationData.length - 1].Traditional;
  const finalAI = simulationData[simulationData.length - 1].GraminRoute;
  const profitIncrease = finalAI - finalTrad;

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
      <div className="flex justify-between items-end mb-6">
        <div>
          <h3 className="font-bold text-lg text-gray-700 flex items-center gap-2">
            <TrendingUp size={20} className="text-blue-600"/> 
            Projected Wealth Growth (60 Days)
          </h3>
          <p className="text-sm text-gray-500">Comparison: Standard vs. GraminRoute Pooling Model</p>
        </div>
        <div className="text-right">
          <p className="text-xs font-bold text-gray-400 uppercase">Net Profit Increase</p>
          <p className="text-3xl font-extrabold text-green-600 flex items-center gap-1">
            +₹{profitIncrease.toLocaleString()} <ArrowUpRight size={28}/>
          </p>
        </div>
      </div>

      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={simulationData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
            <XAxis dataKey="day" hide />
            <YAxis domain={['auto', 'auto']} fontSize={12} tickFormatter={(val) => `₹${val/1000}k`} />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
              formatter={(val) => `₹${val.toLocaleString()}`}
            />
            <Legend />
            {/* Traditional Line */}
            <Line 
              type="monotone" 
              dataKey="Traditional" 
              stroke="#9ca3af" 
              strokeWidth={2} 
              dot={false} 
              strokeDasharray="5 5" 
            />
            {/* GraminRoute Area */}
            <Area 
              type="monotone" 
              dataKey="GraminRoute" 
              stroke="#059669" 
              strokeWidth={3} 
              fill="url(#colorGradient)" 
              fillOpacity={0.1}
            />
            <defs>
              <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#059669" stopOpacity={0.2}/>
                <stop offset="95%" stopColor="#059669" stopOpacity={0}/>
              </linearGradient>
            </defs>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SimulationChart;