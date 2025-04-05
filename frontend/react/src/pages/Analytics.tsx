import React from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';

const Analytics: React.FC = () => {
  return (
    <div className="dashboard-container">
      <Sidebar activePage="analytics" />
      <div className="main-content">
        <Navbar activePage="Analytics" />
        <div className="page-content">
          <h2>Analytics</h2>
          <p>View detailed analytics and insights about your property investments.</p>
          
          <div className="analytics-charts">
            <div className="chart-container">
              <h3>Investment Performance</h3>
              <div className="placeholder-chart"></div>
            </div>
            <div className="chart-container">
              <h3>Risk Distribution</h3>
              <div className="placeholder-chart"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
