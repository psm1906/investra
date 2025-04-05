// frontend/react/src/components/Sidebar.tsx
import React from 'react';

const Sidebar: React.FC = () => {
  return (
    <div className="sidebar">
      <div className="logo-container">
        <div className="house-icon"></div>
      </div>
      
      <div className="sidebar-menu">
        <a href="#" className="menu-item active">
          <span>Property Analysis</span>
        </a>
        <a href="#" className="menu-item">
          <span>Portfolio Dashboard</span>
        </a>
        <a href="#" className="menu-item">
          <span>Market Insights</span>
        </a>
        <a href="#" className="menu-item">
          <span>Investment History</span>
        </a>
        <a href="#" className="menu-item">
          <span>Settings</span>
        </a>
      </div>
      
      <div className="market-trends">
        <h3>Market Trends</h3>
        <div className="trend-item">
          <div className="trend-label">Interest Rates</div>
          <div className="trend-chart interest-chart"></div>
        </div>
        <div className="trend-item">
          <div className="trend-label">Home Prices</div>
          <div className="trend-chart prices-chart"></div>
        </div>
      </div>
      
      <div className="sidebar-footer">
        <p>© 2025 RiskRadar</p>
      </div>
    </div>
  );
};

export default Sidebar;
