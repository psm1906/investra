// frontend/react/src/components/Navbar.tsx
import React from 'react';

const Navbar: React.FC = () => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="app-title">Real Estate Investment Risk Detector</span>
      </div>
      <div className="navbar-menu">
        <a href="#" className="nav-item active">Dashboard</a>
        <a href="#" className="nav-item">Analytics</a>
        <a href="#" className="nav-item">Properties</a>
        <a href="#" className="nav-item">Reports</a>
      </div>
      <div className="navbar-user">
        <div className="user-avatar">JD</div>
      </div>
    </nav>
  );
};

export default Navbar;
