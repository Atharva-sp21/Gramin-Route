import { db } from "./firebaseConfig";
import { collection, addDoc, getDocs, doc, updateDoc, deleteDoc } from "firebase/firestore";

// Save new order (Retailer)
export const saveOrder = async (order) => {
  return await addDoc(collection(db, "pendingOrders"), order);
};

// Update order status (Distributor approves shipment)
export const updateOrderStatus = async (orderId, status) => {
  const orderRef = doc(db, "pendingOrders", orderId);
  await updateDoc(orderRef, { status });
};

// Delete pending order after pooling
export const deletePendingOrder = async (orderId) => {
  await deleteDoc(doc(db, "pendingOrders", orderId));
};

// Store pooled orders
export const storeOrderPool = async (pool) => {
  return await addDoc(collection(db, "orderPools"), pool);
};

// Fetch stock data
export const fetchStock = async () => {
  const snap = await getDocs(collection(db, "retailerStock"));
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
};

// Save stock CSV updates
export const saveStock = async (stockItem) => {
  return await addDoc(collection(db, "retailerStock"), stockItem);
};

// Fetch all pools for distributor
export const fetchPools = async () => {
  const snap = await getDocs(collection(db, "orderPools"));
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
};

// Approve pool shipment
export const approvePoolShipment = async (poolId, status) => {
  const poolRef = doc(db, "orderPools", poolId);
  await updateDoc(poolRef, { status });
};
