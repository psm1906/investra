// src/pages/Dashboard.tsx
import React from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import PropertyAnalysis from '../components/PropertyAnalysis';

const Dashboard: React.FC = () => {
  return (
    <div className="dashboard-container">
      <Sidebar activePage="dashboard" />
      <div className="main-content">
        <Navbar activePage="Dashboard" />
        <PropertyAnalysis />
      </div>
    </div>
  );
};

export default Dashboard;
