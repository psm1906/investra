// src/components/Sidebar.tsx
import React from 'react';
import { Link } from 'react-router-dom';

interface SidebarProps {
  activePage?: string;
}

const Sidebar: React.FC<SidebarProps> = ({ activePage = 'dashboard' }) => {
  return (
    <div className="sidebar">
      <div className="logo-container">
        <Link to="/" className="logo">
          <div className="house-icon"></div>
        </Link>
      </div>
      
      <div className="sidebar-menu">
        <Link to="/" className={`menu-item ${activePage === 'dashboard' ? 'active' : ''}`}>
          <div className="menu-icon property-icon"></div>
          <span>Property Analysis</span>
        </Link>
        <Link to="/properties" className={`menu-item ${activePage === 'properties' ? 'active' : ''}`}>
          <div className="menu-icon dashboard-icon"></div>
          <span>Portfolio Dashboard</span>
        </Link>
        <Link to="/analytics" className={`menu-item ${activePage === 'analytics' ? 'active' : ''}`}>
          <div className="menu-icon insights-icon"></div>
          <span>Market Insights</span>
        </Link>
        <Link to="/reports" className={`menu-item ${activePage === 'reports' ? 'active' : ''}`}>
          <div className="menu-icon history-icon"></div>
          <span>Investment History</span>
        </Link>
        <Link to="/settings" className={`menu-item ${activePage === 'settings' ? 'active' : ''}`}>
          <div className="menu-icon settings-icon"></div>
          <span>Settings</span>
        </Link>
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
