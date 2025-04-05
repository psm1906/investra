// src/components/Navbar.tsx
import React from 'react';
import { Link } from 'react-router-dom';

interface NavbarProps {
  activePage?: string;
}

const Navbar: React.FC<NavbarProps> = ({ activePage = 'Dashboard' }) => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="app-title">Real Estate Investment Risk Detector</span>
      </div>
      <div className="navbar-menu">
        <Link to="/" className={`nav-item ${activePage === 'Dashboard' ? 'active' : ''}`}>
          Dashboard
        </Link>
        <Link to="/analytics" className={`nav-item ${activePage === 'Analytics' ? 'active' : ''}`}>
          Analytics
        </Link>
        <Link to="/properties" className={`nav-item ${activePage === 'Properties' ? 'active' : ''}`}>
          Properties
        </Link>
        <Link to="/reports" className={`nav-item ${activePage === 'Reports' ? 'active' : ''}`}>
          Reports
        </Link>
      </div>
      <div className="navbar-user">
        <div className="user-avatar">JD</div>
      </div>
    </nav>
  );
};

export default Navbar;
