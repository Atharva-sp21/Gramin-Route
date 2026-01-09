import { db } from '../firebase/firebaseConfig';
import { 
  collection, getDocs, doc, updateDoc, addDoc, query, where, writeBatch, deleteDoc 
} from 'firebase/firestore';

// ==========================================
// 1. HARDCODED TELANGANA DATA (100 STORES)
// ==========================================
const KIRANA_DATA = [
  { id: "R001", name: "Sri Balaji Kirana 0", location: "Jangaon District", lat: 17.750854, lon: 79.122778, stock: 162 },
  { id: "R002", name: "Sri Balaji Kirana 1", location: "Jangaon District", lat: 17.714069, lon: 79.224329, stock: 57 },
  { id: "R003", name: "Sri Balaji Kirana 2", location: "Jangaon District", lat: 17.692614, lon: 79.233112, stock: 45 },
  { id: "R004", name: "Sri Balaji Kirana 3", location: "Jangaon District", lat: 17.733689, lon: 79.157468, stock: 156 },
  { id: "R005", name: "Sri Balaji Kirana 4", location: "Jangaon District", lat: 17.663602, lon: 79.206898, stock: 10 },
  { id: "R006", name: "Sri Balaji Kirana 5", location: "Jangaon District", lat: 17.667480, lon: 79.096357, stock: 44 },
  { id: "R007", name: "Sri Balaji Kirana 6", location: "Jangaon District", lat: 17.800985, lon: 79.112704, stock: 149 },
  { id: "R008", name: "Sri Balaji Kirana 7", location: "Jangaon District", lat: 17.835891, lon: 79.184357, stock: 134 },
  { id: "R009", name: "Sri Balaji Kirana 8", location: "Jangaon District", lat: 17.762900, lon: 79.184926, stock: 77 },
  { id: "R010", name: "Sri Balaji Kirana 9", location: "Jangaon District", lat: 17.751378, lon: 79.217296, stock: 57 },
  { id: "R011", name: "Sri Balaji Kirana 10", location: "Jangaon District", lat: 17.632029, lon: 79.232990, stock: 184 },
  { id: "R012", name: "Sri Balaji Kirana 11", location: "Jangaon District", lat: 17.749257, lon: 79.148143, stock: 104 },
  { id: "R013", name: "Sri Balaji Kirana 12", location: "Jangaon District", lat: 17.769666, lon: 79.127749, stock: 154 },
  { id: "R014", name: "Sri Balaji Kirana 13", location: "Jangaon District", lat: 17.680335, lon: 79.137434, stock: 21 },
  { id: "R015", name: "Sri Balaji Kirana 14", location: "Jangaon District", lat: 17.706004, lon: 79.251930, stock: 93 },
  { id: "R016", name: "Sri Balaji Kirana 15", location: "Jangaon District", lat: 17.687115, lon: 79.169524, stock: 36 },
  { id: "R017", name: "Sri Balaji Kirana 16", location: "Jangaon District", lat: 17.762132, lon: 79.218565, stock: 83 },
  { id: "R018", name: "Sri Balaji Kirana 17", location: "Jangaon District", lat: 17.808346, lon: 79.080801, stock: 7 },
  { id: "R019", name: "Sri Balaji Kirana 18", location: "Jangaon District", lat: 17.697667, lon: 79.151367, stock: 131 },
  { id: "R020", name: "Sri Balaji Kirana 19", location: "Jangaon District", lat: 17.697398, lon: 79.185893, stock: 187 },
  { id: "R021", name: "Sri Balaji Kirana 20", location: "Jangaon District", lat: 17.768565, lon: 79.210682, stock: 125 },
  { id: "R022", name: "Sri Balaji Kirana 21", location: "Jangaon District", lat: 17.786090, lon: 79.123006, stock: 11 },
  { id: "R023", name: "Sri Balaji Kirana 22", location: "Jangaon District", lat: 17.719951, lon: 79.151268, stock: 79 },
  { id: "R024", name: "Sri Balaji Kirana 23", location: "Jangaon District", lat: 17.697252, lon: 79.160785, stock: 93 },
  { id: "R025", name: "Sri Balaji Kirana 24", location: "Jangaon District", lat: 17.711261, lon: 79.105262, stock: 42 },
  { id: "R026", name: "Sri Balaji Kirana 25", location: "Jangaon District", lat: 17.685504, lon: 79.235756, stock: 108 },
  { id: "R027", name: "Sri Balaji Kirana 26", location: "Jangaon District", lat: 17.578380, lon: 79.102601, stock: 150 },
  { id: "R028", name: "Sri Balaji Kirana 27", location: "Jangaon District", lat: 17.735529, lon: 79.123400, stock: 121 },
  { id: "R029", name: "Sri Balaji Kirana 28", location: "Jangaon District", lat: 17.668124, lon: 79.260994, stock: 73 },
  { id: "R030", name: "Sri Balaji Kirana 29", location: "Jangaon District", lat: 17.665852, lon: 79.154202, stock: 91 },
  { id: "R031", name: "Sri Balaji Kirana 30", location: "Jangaon District", lat: 17.704672, lon: 79.206425, stock: 146 },
  { id: "R032", name: "Sri Balaji Kirana 31", location: "Jangaon District", lat: 17.798521, lon: 79.122759, stock: 185 },
  { id: "R033", name: "Sri Balaji Kirana 32", location: "Jangaon District", lat: 17.697282, lon: 79.252052, stock: 19 },
  { id: "R034", name: "Sri Balaji Kirana 33", location: "Jangaon District", lat: 17.756688, lon: 79.191410, stock: 170 },
  { id: "R035", name: "Sri Balaji Kirana 34", location: "Jangaon District", lat: 17.755007, lon: 79.070060, stock: 33 },
  { id: "R036", name: "Sri Balaji Kirana 35", location: "Jangaon District", lat: 17.746104, lon: 79.192734, stock: 169 },
  { id: "R037", name: "Sri Balaji Kirana 36", location: "Jangaon District", lat: 17.679429, lon: 79.151874, stock: 186 },
  { id: "R038", name: "Sri Balaji Kirana 37", location: "Jangaon District", lat: 17.661813, lon: 79.125340, stock: 187 },
  { id: "R039", name: "Sri Balaji Kirana 38", location: "Jangaon District", lat: 17.666070, lon: 79.189753, stock: 28 },
  { id: "R040", name: "Sri Balaji Kirana 39", location: "Jangaon District", lat: 17.588356, lon: 79.230448, stock: 27 },
  { id: "R041", name: "Sri Balaji Kirana 40", location: "Jangaon District", lat: 17.730707, lon: 79.132133, stock: 39 },
  { id: "R042", name: "Sri Balaji Kirana 41", location: "Jangaon District", lat: 17.730091, lon: 79.083276, stock: 118 },
  { id: "R043", name: "Sri Balaji Kirana 42", location: "Jangaon District", lat: 17.821649, lon: 79.143905, stock: 77 },
  { id: "R044", name: "Sri Balaji Kirana 43", location: "Jangaon District", lat: 17.712675, lon: 79.148316, stock: 52 },
  { id: "R045", name: "Sri Balaji Kirana 44", location: "Jangaon District", lat: 17.698100, lon: 79.184277, stock: 71 },
  { id: "R046", name: "Sri Balaji Kirana 45", location: "Jangaon District", lat: 17.700454, lon: 79.184554, stock: 51 },
  { id: "R047", name: "Sri Balaji Kirana 46", location: "Jangaon District", lat: 17.709105, lon: 79.131703, stock: 123 },
  { id: "R048", name: "Sri Balaji Kirana 47", location: "Jangaon District", lat: 17.707832, lon: 79.112625, stock: 103 },
  { id: "R049", name: "Sri Balaji Kirana 48", location: "Jangaon District", lat: 17.753305, lon: 79.140546, stock: 41 },
  { id: "R050", name: "Sri Balaji Kirana 49", location: "Jangaon District", lat: 17.740921, lon: 79.220302, stock: 89 },
  { id: "R051", name: "Sri Balaji Kirana 50", location: "Jangaon District", lat: 17.606749, lon: 79.150320, stock: 137 },
  { id: "R052", name: "Sri Balaji Kirana 51", location: "Jangaon District", lat: 17.744489, lon: 79.145062, stock: 171 },
  { id: "R053", name: "Sri Balaji Kirana 52", location: "Jangaon District", lat: 17.731544, lon: 79.251108, stock: 32 },
  { id: "R054", name: "Sri Balaji Kirana 53", location: "Jangaon District", lat: 17.793774, lon: 79.078332, stock: 72 },
  { id: "R055", name: "Sri Balaji Kirana 54", location: "Jangaon District", lat: 17.662053, lon: 79.082071, stock: 184 },
  { id: "R056", name: "Sri Balaji Kirana 55", location: "Jangaon District", lat: 17.737339, lon: 79.144788, stock: 104 },
  { id: "R057", name: "Sri Balaji Kirana 56", location: "Jangaon District", lat: 17.742348, lon: 79.093603, stock: 44 },
  { id: "R058", name: "Sri Balaji Kirana 57", location: "Jangaon District", lat: 17.757023, lon: 79.222072, stock: 92 },
  { id: "R059", name: "Sri Balaji Kirana 58", location: "Jangaon District", lat: 17.742146, lon: 79.203230, stock: 19 },
  { id: "R060", name: "Sri Balaji Kirana 59", location: "Jangaon District", lat: 17.735738, lon: 79.128920, stock: 153 },
  { id: "R061", name: "Sri Balaji Kirana 60", location: "Jangaon District", lat: 17.666631, lon: 79.121459, stock: 35 },
  { id: "R062", name: "Sri Balaji Kirana 61", location: "Jangaon District", lat: 17.643842, lon: 79.184386, stock: 64 },
  { id: "R063", name: "Sri Balaji Kirana 62", location: "Jangaon District", lat: 17.725155, lon: 79.127704, stock: 21 },
  { id: "R064", name: "Sri Balaji Kirana 63", location: "Jangaon District", lat: 17.746501, lon: 79.135899, stock: 15 },
  { id: "R065", name: "Sri Balaji Kirana 64", location: "Jangaon District", lat: 17.693680, lon: 79.157894, stock: 99 },
  { id: "R066", name: "Sri Balaji Kirana 65", location: "Jangaon District", lat: 17.725404, lon: 79.191321, stock: 145 },
  { id: "R067", name: "Sri Balaji Kirana 66", location: "Jangaon District", lat: 17.663731, lon: 79.194534, stock: 188 },
  { id: "R068", name: "Sri Balaji Kirana 67", location: "Jangaon District", lat: 17.721417, lon: 79.134482, stock: 134 },
  { id: "R069", name: "Sri Balaji Kirana 68", location: "Jangaon District", lat: 17.777718, lon: 79.137672, stock: 114 },
  { id: "R070", name: "Sri Balaji Kirana 69", location: "Jangaon District", lat: 17.676209, lon: 79.107525, stock: 97 },
  { id: "R071", name: "Sri Balaji Kirana 70", location: "Jangaon District", lat: 17.696359, lon: 79.217888, stock: 32 },
  { id: "R072", name: "Sri Balaji Kirana 71", location: "Jangaon District", lat: 17.800638, lon: 79.081157, stock: 169 },
  { id: "R073", name: "Sri Balaji Kirana 72", location: "Jangaon District", lat: 17.752442, lon: 79.213062, stock: 184 },
  { id: "R074", name: "Sri Balaji Kirana 73", location: "Jangaon District", lat: 17.814523, lon: 79.175695, stock: 138 },
  { id: "R075", name: "Sri Balaji Kirana 74", location: "Jangaon District", lat: 17.777834, lon: 79.181612, stock: 8 },
  { id: "R076", name: "Sri Balaji Kirana 75", location: "Jangaon District", lat: 17.695114, lon: 79.214893, stock: 189 },
  { id: "R077", name: "Sri Balaji Kirana 76", location: "Jangaon District", lat: 17.689460, lon: 79.148742, stock: 93 },
  { id: "R078", name: "Sri Balaji Kirana 77", location: "Jangaon District", lat: 17.775440, lon: 79.189758, stock: 60 },
  { id: "R079", name: "Sri Balaji Kirana 78", location: "Jangaon District", lat: 17.708035, lon: 79.175633, stock: 27 },
  { id: "R080", name: "Sri Balaji Kirana 79", location: "Jangaon District", lat: 17.664686, lon: 79.217750, stock: 140 },
  { id: "R081", name: "Sri Balaji Kirana 80", location: "Jangaon District", lat: 17.691752, lon: 79.114171, stock: 109 },
  { id: "R082", name: "Sri Balaji Kirana 81", location: "Jangaon District", lat: 17.652280, lon: 79.108889, stock: 76 },
  { id: "R083", name: "Sri Balaji Kirana 82", location: "Jangaon District", lat: 17.644091, lon: 79.163166, stock: 123 },
  { id: "R084", name: "Sri Balaji Kirana 83", location: "Jangaon District", lat: 17.728567, lon: 79.138167, stock: 5 },
  { id: "R085", name: "Sri Balaji Kirana 84", location: "Jangaon District", lat: 17.708293, lon: 79.191817, stock: 131 },
  { id: "R086", name: "Sri Balaji Kirana 85", location: "Jangaon District", lat: 17.729532, lon: 79.199064, stock: 57 },
  { id: "R087", name: "Sri Balaji Kirana 86", location: "Jangaon District", lat: 17.708513, lon: 79.173193, stock: 98 },
  { id: "R088", name: "Sri Balaji Kirana 87", location: "Jangaon District", lat: 17.688456, lon: 79.205543, stock: 65 },
  { id: "R089", name: "Sri Balaji Kirana 88", location: "Jangaon District", lat: 17.688962, lon: 79.170784, stock: 79 },
  { id: "R090", name: "Sri Balaji Kirana 89", location: "Jangaon District", lat: 17.723785, lon: 79.105754, stock: 20 },
  { id: "R091", name: "Sri Balaji Kirana 90", location: "Jangaon District", lat: 17.661111, lon: 79.126661, stock: 121 },
  { id: "R092", name: "Sri Balaji Kirana 91", location: "Jangaon District", lat: 17.782332, lon: 79.195718, stock: 76 },
  { id: "R093", name: "Sri Balaji Kirana 92", location: "Jangaon District", lat: 17.706713, lon: 79.151509, stock: 90 },
  { id: "R094", name: "Sri Balaji Kirana 93", location: "Jangaon District", lat: 17.777952, lon: 79.195877, stock: 64 },
  { id: "R095", name: "Sri Balaji Kirana 94", location: "Jangaon District", lat: 17.722513, lon: 79.101981, stock: 102 },
  { id: "R096", name: "Sri Balaji Kirana 95", location: "Jangaon District", lat: 17.718446, lon: 79.193507, stock: 5 },
  { id: "R097", name: "Sri Balaji Kirana 96", location: "Jangaon District", lat: 17.832144, lon: 79.185466, stock: 34 },
  { id: "R098", name: "Sri Balaji Kirana 97", location: "Jangaon District", lat: 17.724691, lon: 79.226235, stock: 170 },
  { id: "R099", name: "Sri Balaji Kirana 98", location: "Jangaon District", lat: 17.718785, lon: 79.208840, stock: 24 },
  { id: "R100", name: "Sri Balaji Kirana 99", location: "Jangaon District", lat: 17.715496, lon: 79.195881, stock: 87 }
];

const CATALOG = [
  { id: 'P001', name: 'Rice (50kg)', category: 'Grains', unit_price: 2500, reorder_level: 10 },
  { id: 'P002', name: 'Wheat Flour (40kg)', category: 'Grains', unit_price: 1800, reorder_level: 15 },
  { id: 'P003', name: 'Sugar (50kg)', category: 'Sweeteners', unit_price: 2200, reorder_level: 12 },
  { id: 'P004', name: 'Cooking Oil (15L)', category: 'Oils', unit_price: 1500, reorder_level: 20 },
  { id: 'P005', name: 'Pulses Mix (25kg)', category: 'Pulses', unit_price: 3000, reorder_level: 8 },
];

// ==========================================
// 2. SEED FUNCTION (Deletes Old Data First)
// ==========================================
export const seedDatabase = async () => {
  console.log("🌱 Starting Clean Seed Process...");

  // 1. DELETE OLD DATA (Cleanup Step)
  try {
      const usersSnap = await getDocs(collection(db, "users"));
      usersSnap.forEach(async (doc) => await deleteDoc(doc.ref));
      
      const ordersSnap = await getDocs(collection(db, "orders"));
      ordersSnap.forEach(async (doc) => await deleteDoc(doc.ref));
      console.log("🗑️ Old Data Cleared.");
  } catch (e) {
      console.log("⚠️ Cleanup skipped (first run or permission issue). Continuing...");
  }

  // 2. BATCH WRITE NEW DATA
  const batch = writeBatch(db);

  // A. Create Distributor (D001) in Jangaon
  const d001Ref = doc(collection(db, "users"), "user_D001");
  batch.set(d001Ref, {
    id: 'D001', 
    type: 'distributor', 
    name: 'GraminRoute Hub', 
    location: 'Jangaon District', 
    lat: 17.7200, // Center of Jangaon
    lon: 79.1600,
    password: 'dist123' 
  });

  // B. Create 100 Retailers & Orders
  KIRANA_DATA.forEach(retailer => {
    // 1. Create Retailer User Doc
    const userRef = doc(collection(db, "users"), `user_${retailer.id}`);
    batch.set(userRef, {
        id: retailer.id,
        type: 'retailer',
        name: retailer.name,
        location: retailer.location,
        lat: retailer.lat,
        lon: retailer.lon,
        stock: retailer.stock,
        password: retailer.id === 'R001' ? 'sharma123' : 'pass' 
    });

    // 2. If R001 (Our Hero), Create History
    if (retailer.id === 'R001') {
        // Create Inventory for AI Recommendations
        CATALOG.forEach(product => {
            const invRef = doc(collection(db, "inventory"));
            batch.set(invRef, {
                ...product,
                retailer_id: 'R001',
                current_stock: Math.floor(Math.random() * 20),
                last_updated: new Date().toISOString().split('T')[0]
            });
        });
    } 
    // 3. For R002-R100, Create Pending Orders (Pooling Data)
    else {
        // 50% chance to have a pending order
        if (Math.random() > 0.5) {
            const product = CATALOG[Math.floor(Math.random() * CATALOG.length)];
            const qty = Math.floor(Math.random() * 10) + 10; 
            const orderRef = doc(collection(db, "orders"));
            batch.set(orderRef, {
                retailer_id: retailer.id,
                retailer_name: retailer.name,
                retailer_location: retailer.location,
                retailer_lat: retailer.lat, // KEY for Python Pooling
                retailer_lon: retailer.lon,
                product_id: product.id,
                product_name: product.name,
                quantity: qty,
                unit_price: product.unit_price,
                total_amount: qty * product.unit_price,
                status: 'pending',
                date: new Date().toISOString().split('T')[0],
                source: 'Auto-Restock'
            });
        }
    }
  });

  await batch.commit();
  console.log("✅ SEED COMPLETE: 100 Telangana Retailers Created.");
  alert("Database updated with 100 Telangana Stores! Login as D001.");
};

// ==========================================
// 3. API FUNCTIONS (AUTH, ORDERS, AI)
// ==========================================
const PYTHON_API_URL = "http://localhost:8000";

// --- Users ---
export const loginUser = async (userId, password, type) => {
  const usersRef = collection(db, "users");
  const q = query(usersRef, where("id", "==", userId), where("password", "==", password), where("type", "==", type));
  const snapshot = await getDocs(q);
  if (snapshot.empty) return null;
  return { docId: snapshot.docs[0].id, ...snapshot.docs[0].data() };
};

export const fetchAllRetailers = async () => {
    const q = query(collection(db, "users"), where("type", "==", "retailer"));
    const snapshot = await getDocs(q);
    return snapshot.docs.map(doc => doc.data());
};

// --- Inventory ---
export const fetchInventory = async (retailerId) => {
  const q = query(collection(db, "inventory"), where("retailer_id", "==", retailerId));
  const snapshot = await getDocs(q);
  return snapshot.docs.map(doc => ({ docId: doc.id, ...doc.data() }));
};

// --- Orders ---
export const fetchOrders = async (retailerId = null) => {
  let q;
  if (retailerId) {
    q = query(collection(db, "orders"), where("retailer_id", "==", retailerId));
  } else {
    q = query(collection(db, "orders")); 
  }
  const snapshot = await getDocs(q);
  return snapshot.docs.map(doc => ({ docId: doc.id, ...doc.data() }));
};

export const createOrder = async (orderData) => {
  await addDoc(collection(db, "orders"), orderData);
};

export const updateOrderStatus = async (docId, status) => {
  const orderRef = doc(db, "orders", docId);
  await updateDoc(orderRef, { status: status });
};

// --- AI BRAIN INTEGRATION ---
export const getAIRecommendation = async (retailerData) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/recommend_distributor`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        shop_id: retailerData.id,
        lat: retailerData.lat,
        lon: retailerData.lon,
        current_stock: retailerData.current_stock,
        is_festival: false 
      })
    });
    return await response.json();
  } catch (error) {
    console.error("AI Brain Offline:", error);
    return null; 
  }
};

export const getOptimizedPools = async (orders) => {
  const pythonPayload = orders.map(o => ({
    shop_id: o.retailer_id, 
    lat: o.retailer_lat || 17.72, // Default to Jangaon if missing
    lon: o.retailer_lon || 79.16,
    qty_needed: o.quantity
  }));

  try {
    const response = await fetch(`${PYTHON_API_URL}/pool_orders`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pythonPayload)
    });
    return await response.json();
  } catch (error) {
    console.error("Pooling Engine Offline:", error);
    return [];
  }
};