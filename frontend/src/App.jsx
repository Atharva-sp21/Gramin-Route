import React, { useState } from 'react';

// Import Pages
import LoginPage from './pages/Login';
import RetailerDashboard from './pages/RetailerDashboard';
import DistributorDashboard from './pages/DistributorDashboard';

export default function App() {
  // --- STATE MANAGEMENT ---
  // Stores the type of user ('retailer' or 'distributor')
  const [userType, setUserType] = useState(null);
  
  // Stores the full user object (id, name, location, lat, lon) from Firestore
  const [currentUser, setCurrentUser] = useState(null);

  // --- HANDLERS ---
  
  // Called when the Login Component successfully authenticates a user
  const handleLogin = (type, user) => {
    console.log(`✅ Logged in as ${type}:`, user.name);
    setUserType(type);
    setCurrentUser(user);
  };

  // Called when the Logout button is clicked in either dashboard
  const handleLogout = () => {
    console.log("👋 Logging out...");
    setUserType(null);
    setCurrentUser(null);
    // Optional: Clear any other session storage here if you add it later
  };

  // --- CONDITIONAL RENDERING ---

  // 1. If no user is logged in, show the Login Page
  if (!userType || !currentUser) {
    return <LoginPage onLogin={handleLogin} />;
  }

  // 2. If the user is a Retailer, show the Retailer Dashboard
  // We pass the user object (for data fetching) and the logout handler
  if (userType === 'retailer') {
    return (
      <RetailerDashboard 
        user={currentUser} 
        onLogout={handleLogout} 
      />
    );
  }

  // 3. If the user is a Distributor, show the Distributor Dashboard
  return (
    <DistributorDashboard 
      user={currentUser} 
      onLogout={handleLogout} 
    />
  );
}