import React, { useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, Marker, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default Leaflet marker icons
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// Custom Truck Icon
const truckIcon = new L.Icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/71/71222.png', 
    iconSize: [35, 35],
    className: 'hub-icon'
});

const LogisticsMap = ({ retailers = [], pools = [] }) => {
  // Center logic
  const defaultCenter = [17.72, 79.16]; 
  const mapCenter = (retailers.length > 0 && retailers[0].lat) 
    ? [parseFloat(retailers[0].lat), parseFloat(retailers[0].lon)] 
    : defaultCenter;

  // --- PRE-CALCULATE VALID LINES ---
  const validPolylines = useMemo(() => {
    const lines = [];
    console.log(`🗺️ Map Debug: Processing ${pools.length} pools...`);

    if (!pools || !Array.isArray(pools)) return lines;

    pools.forEach(pool => {
        if (!pool.orders || pool.orders.length === 0) return;

        // 1. Determine Pool Anchor (Center)
        // Try Python center first, otherwise default to first shop in the pool
        let anchorLat = parseFloat(pool.center_lat);
        let anchorLon = parseFloat(pool.center_lon);

        if (isNaN(anchorLat) || isNaN(anchorLon)) {
            // Fallback: Use the first order's location as the anchor
            anchorLat = parseFloat(pool.orders[0].retailer_lat);
            anchorLon = parseFloat(pool.orders[0].retailer_lon);
            console.log(`⚠️ Pool ${pool.pool_id} missing center. Using first shop as anchor.`);
        }

        // 2. Draw line from Anchor to every shop in the pool
        pool.orders.forEach((order, idx) => {
            const endLat = parseFloat(order.retailer_lat);
            const endLon = parseFloat(order.retailer_lon);

            // Validity Check
            const isValid = 
                !isNaN(anchorLat) && !isNaN(anchorLon) && 
                !isNaN(endLat) && !isNaN(endLon);

            if (isValid) {
                // Check if Wholesale
                // Check multiple ways the backend might send this flag
                const isWholesale = 
                    (pool.discount && pool.discount.includes('WHOLESALE')) ||
                    pool.total_qty > 50; 

                lines.push({
                    key: `${pool.pool_id}-${order.id || idx}`,
                    positions: [[anchorLat, anchorLon], [endLat, endLon]],
                    color: isWholesale ? '#00FF00' : '#FFA500' // Green vs Orange
                });
            } else {
                console.warn("❌ Invalid Coords for order:", order.id, {anchorLat, anchorLon, endLat, endLon});
            }
        });
    });
    console.log(`✅ Generated ${lines.length} route lines.`);
    return lines;
  }, [pools]);

  return (
    <div className="h-[500px] w-full rounded-xl overflow-hidden shadow-lg border border-gray-700 relative">
      <MapContainer key={mapCenter.toString()} center={mapCenter} zoom={11} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
        
        <TileLayer
          attribution='&copy; CARTO'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* Hub Marker */}
        <Marker position={defaultCenter} icon={truckIcon}>
            <Tooltip permanent direction="top" className="font-bold">HUB (D001)</Tooltip>
        </Marker>

        {/* Render Pre-Calculated Lines */}
        {validPolylines.map(line => (
            <Polyline 
                key={line.key}
                positions={line.positions}
                pathOptions={{ 
                    color: line.color, 
                    weight: 3, // Thicker lines
                    dashArray: '10, 10', 
                    opacity: 0.8 // More visible
                }} 
            />
        ))}

        {/* Retailer Nodes */}
        {retailers.map((retailer) => {
            const lat = parseFloat(retailer.lat);
            const lon = parseFloat(retailer.lon);
            
            if (isNaN(lat) || isNaN(lon)) return null;

            const isCritical = retailer.stock < 20; 
            const color = isCritical ? '#FF0044' : '#00DDFF'; 

            return (
                <CircleMarker
                    key={retailer.id}
                    center={[lat, lon]}
                    pathOptions={{ color: color, fillColor: color, fillOpacity: 0.7 }}
                    radius={isCritical ? 6 : 4}
                >
                    <Popup>
                        <div className="text-sm font-sans">
                            <strong>{retailer.name}</strong><br/>
                            Stock: <span style={{color: isCritical ? 'red' : 'green'}}>{retailer.stock}</span>
                        </div>
                    </Popup>
                </CircleMarker>
            );
        })}

      </MapContainer>
      
      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-gray-900/90 p-3 rounded text-white text-xs z-[1000] border border-gray-600">
          <div className="flex items-center gap-2 mb-1"><div className="w-3 h-3 rounded-full bg-[#FF0044]"></div> Critical Stock</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-3 h-3 rounded-full bg-[#00DDFF]"></div> Healthy Stock</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-6 h-0.5 bg-[#00FF00] border border-dashed border-green-500"></div> Wholesale Route (&gt;50 units)</div>
          <div className="flex items-center gap-2"><div className="w-6 h-0.5 bg-[#FFA500] border border-dashed border-orange-500"></div> Standard Route</div>
      </div>
    </div>
  );
};

export default LogisticsMap;