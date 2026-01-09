import React, { useState } from 'react';
import { Plus } from 'lucide-react';

const ProductCard = ({ product, onAddToCart }) => {
  const [quantity, setQuantity] = useState(1);

  return (
    <div className="border rounded-lg p-4 hover:shadow-lg transition bg-white">
      <div className="text-center mb-3">
        <span className="text-5xl">{product.image}</span>
      </div>
      <h3 className="font-bold text-gray-800 mb-1">{product.name}</h3>
      <p className="text-sm text-gray-600 mb-2">{product.category}</p>
      <p className="text-2xl font-bold text-green-600 mb-3">₹{product.unit_price.toLocaleString()}</p>
      <div className="flex gap-2 items-center mb-3 justify-center">
        <button 
          onClick={() => setQuantity(Math.max(1, quantity - 1))}
          className="bg-gray-200 px-3 py-1 rounded hover:bg-gray-300"
        > - </button>
        <input 
          type="number" 
          value={quantity}
          onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))}
          className="w-16 text-center border rounded py-1"
        />
        <button 
          onClick={() => setQuantity(quantity + 1)}
          className="bg-gray-200 px-3 py-1 rounded hover:bg-gray-300"
        > + </button>
      </div>
      <button
        onClick={() => onAddToCart(product, quantity)}
        className="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 font-semibold flex items-center justify-center gap-2"
      >
        <Plus size={16} /> Add to Cart
      </button>
    </div>
  );
};

export default ProductCard;