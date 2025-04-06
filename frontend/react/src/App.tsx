import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { SignedIn, SignedOut, UserButton, SignInButton } from '@clerk/clerk-react';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Properties from './pages/Properties';
import Reports from './pages/Reports';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Footer from './components/Footer';

function App() {
  return (
    <div className="dashboard-container">
      <Navbar />
      <Sidebar />
      
      <div className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/properties" element={<Properties />} />
          <Route path="/reports" element={<Reports />} />
        </Routes>
      </div>
      
      <Footer />
    </div>
  );
}

export default App;
