// src/pages/Dashboard.tsx
import React from 'react';
import PropertyAnalysis from '../components/PropertyAnalysis';

const Dashboard: React.FC = () => {
  return (
    <div className="dashboard-container">
      <div className="main-content">
        <PropertyAnalysis />
      </div>
    </div>
  );
};

export default Dashboard;
