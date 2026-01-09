import React, { useState } from 'react';
import { loginUser, seedDatabase } from '../services/api';

const LoginPage = ({ onLogin }) => {
  const [userType, setUserType] = useState('retailer');
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setLoading(true);
    setError('');
    try {
      const user = await loginUser(userId, password, userType);
      if (user) {
        onLogin(userType, user);
      } else {
        setError('Invalid credentials or user not found');
      }
    } catch (err) {
      console.error(err);
      setError('Login failed. Check console.');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-green-700 mb-2">GraminRoute</h1>
          <p className="text-gray-600">B2B Supply Chain Platform</p>
        </div>

        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setUserType('retailer')}
            className={`flex-1 py-3 rounded-lg font-semibold transition ${
              userType === 'retailer' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >Retailer</button>
          <button
            onClick={() => setUserType('distributor')}
            className={`flex-1 py-3 rounded-lg font-semibold transition ${
              userType === 'distributor' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >Distributor</button>
        </div>

        <div className="space-y-4">
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
            placeholder={userType === 'retailer' ? 'R001' : 'D001'}
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
            placeholder="Password"
          />
          {error && <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm">{error}</div>}
          
          <button
            onClick={handleLogin}
            disabled={loading}
            className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition disabled:bg-gray-400"
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </div>
        
        <div className="mt-8 pt-4 border-t text-center">
            <p className="text-xs text-gray-500 mb-2">First time running this with Firestore?</p>
            <button onClick={seedDatabase} className="text-xs text-blue-600 underline">
                Click here to Seed Database (Create Test Users)
            </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;