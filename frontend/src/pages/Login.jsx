import React, { useState } from 'react';
import { loginUser, seedDatabase } from '../services/api';

const LoginPage = ({ onLogin }) => {
  const [userType, setUserType] = useState('retailer');
  const [userId, setUserId]     = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  const handleLogin = async () => {
    if (!userId || !password) { setError('Enter ID and password'); return; }
    setLoading(true);
    setError('');
    try {
      const user = await loginUser(userId, password, userType);
      if (user) {
        onLogin(userType, user);
      } else {
        setError('Invalid credentials. Check ID and password.');
      }
    } catch (err) {
      console.error(err);
      setError('Login failed. Is the gateway running on port 8080?');
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
          {['retailer', 'distributor'].map(t => (
            <button
              key={t}
              onClick={() => setUserType(t)}
              className={`flex-1 py-3 rounded-lg font-semibold capitalize transition ${
                userType === t
                  ? t === 'retailer' ? 'bg-green-600 text-white' : 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >{t}</button>
          ))}
        </div>

        <div className="space-y-4">
          <input
            type="text"
            value={userId}
            onChange={e => setUserId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleLogin()}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
            placeholder={userType === 'retailer' ? 'R001' : 'D001'}
          />
          <input
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleLogin()}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
            placeholder={userType === 'retailer' ? 'sharma123' : 'dist123'}
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

        <div className="mt-8 pt-4 border-t text-center space-y-2">
          <p className="text-xs text-gray-500">Default credentials after seeding:</p>
          <p className="text-xs font-mono text-gray-700">Retailer: R001 / sharma123</p>
          <p className="text-xs font-mono text-gray-700">Distributor: D001 / dist123</p>
          <button onClick={seedDatabase} className="text-xs text-blue-600 underline mt-2 block mx-auto">
            How to seed the database
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
