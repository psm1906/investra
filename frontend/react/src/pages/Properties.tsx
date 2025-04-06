import React from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';

const Properties: React.FC = () => {
  return (
    <div className="dashboard-container">
      <Sidebar activePage="properties" />
      <div className="main-content">
        <Navbar activePage="Properties" />
        <div className="page-content">
          <h2>Properties</h2>
          <p>Manage your property portfolio</p>
          
          <div className="properties-list">
            <div className="property-card">
              <div className="property-image"></div>
              <div className="property-details">
                <h3>123 Main St, Austin, TX</h3>
                <p>Single-Family Home • $475,000</p>
                <div className="risk-badge medium">Medium Risk (68)</div>
              </div>
            </div>
            
            <div className="property-card">
              <div className="property-image"></div>
              <div className="property-details">
                <h3>456 Oak Ave, Dallas, TX</h3>
                <p>Townhouse • $325,000</p>
                <div className="risk-badge low">Low Risk (32)</div>
              </div>
            </div>
            
            <div className="add-property-card">
              <div className="add-icon">+</div>
              <p>Add New Property</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Properties;
