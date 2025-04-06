import React from 'react';
import Navbar from '../components/Navbar'; // Header component
import Sidebar from '../components/Sidebar'; // Sidebar component

const Analytics: React.FC = () => {
  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <Sidebar activePage="analytics" />

      {/* Main Content */}
      <div className="main-content">
        <Navbar activePage="Analytics" />
        <div className="page-content">
          <h2>Analytics</h2>
          <p>Gain insights into your investments and risk management strategies.</p>

          {/* Analytics Content */}
          <div className="analytics-charts">
            {/* Investment Distribution Section */}
            <div className="chart-container">
              <h3>Investment Distribution</h3>
              <p>Analyze how your investments are distributed across various categories.</p>
              <div className="placeholder-chart"></div> {/* Placeholder for future charts */}
            </div>

            {/* Risk Management Section */}
            <div className="chart-container">
              <h3>Risk Management</h3>
              <p>Understand the risk levels associated with your portfolio.</p>
              <div className="placeholder-chart"></div> {/* Placeholder for future charts */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
