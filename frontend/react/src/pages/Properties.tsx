// src/pages/Properties.tsx
import React, { useEffect, useState } from 'react';
import { useAuth } from '@clerk/clerk-react';

interface PropertyData {
  address: string;
  propertyType: string;
  price: number;
  riskScore: number;
}

const Properties: React.FC = () => {
  // We'll fetch the user’s property list from the backend
  const [properties, setProperties] = useState<PropertyData[]>([]);

  // Clerk's useAuth hook: can get the userId, signOut, and (importantly) a JWT token
  const { isLoaded, userId, getToken } = useAuth();

  // On mount (and whenever user changes), fetch property list
  useEffect(() => {
    if (!isLoaded) return;      // still loading Clerk
    if (!userId) return;        // user is not signed in

    // Attempt to retrieve a Clerk session token
    getToken({ template: "default" })  // or no args, if you haven't set up a template
      .then((token) => {
        // Now call our backend with an Authorization header
        fetch("/api/properties", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        })
          .then((res) => {
            if (!res.ok) {
              throw new Error("Failed to fetch user properties");
            }
            return res.json();
          })
          .then((data) => {
            console.log("Properties from backend:", data);
            setProperties(data);
          })
          .catch((err) => {
            console.error(err);
          });
      });
  }, [isLoaded, userId, getToken]);

  if (!isLoaded) {
    return <p>Loading Clerk...</p>;
  }

  if (!userId) {
    return (
      <div className="page-content">
        <h2>Please sign in to view properties.</h2>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="main-content">
        <div className="page-content">
          <h2>Properties</h2>
          <p>Manage your property portfolio</p>
          
          <div className="properties-list">
            {properties.map((prop, idx) => {
              // You might have an ID for the property in real code
              const riskClass = prop.riskScore >= 70
                ? "high"
                : prop.riskScore >= 40
                ? "medium"
                : "low";
              return (
                <div className="property-card" key={idx}>
                  <div className="property-image"></div>
                  <div className="property-details">
                    <h3>{prop.address}</h3>
                    <p>{prop.propertyType} • ${prop.price}</p>
                    <div className={`risk-badge ${riskClass}`}>
                      {riskClass.charAt(0).toUpperCase() + riskClass.slice(1)} Risk ({prop.riskScore})
                    </div>
                  </div>
                </div>
              );
            })}
            
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