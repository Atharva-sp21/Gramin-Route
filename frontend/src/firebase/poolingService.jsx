import { db } from "./firebaseConfig";
import { collection, getDocs, addDoc, deleteDoc, doc } from "firebase/firestore";

/**
 * Fetch pending orders, group by product, send to FastAPI pooling,
 * and store pools back in Firestore. Also cleans pending orders after pooling.
 */
export async function poolOrdersAndStore() {
  const snap = await getDocs(collection(db, "pendingOrders"));
  const orders = snap.docs.map(d => ({ id: d.id, ...d.data() }));

  // Group orders by product type
  const grouped = Object.groupBy(orders, o => o.product_id);

  const createdPools = [];

  for (const productId in grouped) {
    const batch = grouped[productId];

    // Call your FastAPI pooling endpoint
    const res = await fetch("http://localhost:8000/pool_orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(batch)
    });

    const data = await res.json();

    // Store each pool returned by API into Firestore
    for (const pool of data.optimized_pools) {
      const poolDoc = {
        pool_id: pool.pool_id,
        product_id: productId,
        product_name: batch[0].product_name, // all same product in this batch
        shops: pool.shops,
        total_qty: pool.total_qty,
        center_lat: pool.center_lat,
        center_lon: pool.center_lon,
        discount_applied: pool.discount_applied,
        status: "awaiting_approval",
        created_at: new Date().toISOString(),
        invoice_generated: false
      };

      const saved = await addDoc(collection(db, "orderPools"), poolDoc);
      createdPools.push({ id: saved.id, ...poolDoc });
    }

    // Delete pending orders that were pooled
    for (const order of batch) {
      await deletePendingOrder(order.id);
    }
  }

  return createdPools;
}

// Helper: delete a pending order
async function deletePendingOrder(orderId) {
  await deleteDoc(doc(db, "pendingOrders", orderId));
}
