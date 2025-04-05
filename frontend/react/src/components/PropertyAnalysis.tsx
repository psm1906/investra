// frontend/react/src/components/PropertyAnalysis.tsx
import React, { useState } from 'react';
import RiskMeter from './RiskMeter';
import RiskFactors from './RiskFactors';

interface PropertyDetails {
  location: string;
  propertyType: string;
  price: number;
  squareFootage: number;
  bedrooms: number;
  bathrooms: number;
  yearBuilt: number;
  condition: string;
}

const PropertyAnalysis: React.FC = () => {
  const [propertyDetails, setPropertyDetails] = useState<PropertyDetails>({
    location: '123 Main St, Austin, TX 78701',
    propertyType: 'Single-Family Home',
    price: 475000,
    squareFootage: 2350,
    bedrooms: 4,
    bathrooms: 3,
    yearBuilt: 2005,
    condition: 'Good'
  });
  
  const [riskScore, setRiskScore] = useState(68);
  const [riskFactors, setRiskFactors] = useState([
    { name: 'Market Volatility', score: 70 },
    { name: 'Interest Rate Exposure', score: 80 },
    { name: 'Neighborhood Growth', score: 55 },
    { name: 'Property Condition Risk', score: 50 }
  ]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setPropertyDetails({
      ...propertyDetails,
      [name]: value
    });
  };

  const analyzeRisk = () => {
    console.log('Analyzing risk for property:', propertyDetails);
  };

  return (
    <div className="property-analysis-container">
      <h2>Property Risk Analysis</h2>
      <p>Input property details to generate comprehensive risk assessment</p>
      
      <div className="analysis-content">
        <div className="property-details-section">
          <div className="section-header">
            <div className="house-icon"></div>
            <h3>Property Details</h3>
          </div>
          
          <div className="form-group">
            <label>Location</label>
            <div className="location-input">
              <input 
                type="text" 
                name="location" 
                value={propertyDetails.location} 
                onChange={handleInputChange}
              />
              <button className="location-btn">
                <div className="map-icon"></div>
              </button>
            </div>
          </div>
          
          <div className="property-image-section">
            <div className="image-placeholder">
              <div className="house-image"></div>
            </div>
            <div className="image-upload-info">
              <h4>Property Images</h4>
              <p>Upload photos or drag and drop files here</p>
            </div>
          </div>
          
          <div className="form-row">
            <div className="form-group half">
              <label>Property Type</label>
              <select 
                name="propertyType" 
                value={propertyDetails.propertyType} 
                onChange={handleInputChange}
              >
                <option>Single-Family Home</option>
                <option>Condo</option>
                <option>Townhouse</option>
                <option>Multi-Family</option>
              </select>
            </div>
            <div className="form-group half">
              <label>Square Footage</label>
              <input 
                type="text" 
                name="squareFootage" 
                value={propertyDetails.squareFootage} 
                onChange={handleInputChange}
              />
            </div>
          </div>
          
          <div className="form-row">
            <div className="form-group half">
              <label>Price / Market Value</label>
              <div className="price-input">
                <span className="currency">$</span>
                <input 
                  type="text" 
                  name="price" 
                  value={propertyDetails.price} 
                  onChange={handleInputChange}
                />
              </div>
            </div>
            <div className="form-group half">
              <label>Year Built</label>
              <input 
                type="text" 
                name="yearBuilt" 
                value={propertyDetails.yearBuilt} 
                onChange={handleInputChange}
              />
            </div>
          </div>
          
          <div className="form-row">
            <div className="form-group half">
              <label>Bedrooms & Bathrooms</label>
              <div className="beds-baths-inputs">
                <input 
                  type="text" 
                  name="bedrooms" 
                  value={propertyDetails.bedrooms} 
                  onChange={handleInputChange}
                  placeholder="Beds"
                />
                <input 
                  type="text" 
                  name="bathrooms" 
                  value={propertyDetails.bathrooms} 
                  onChange={handleInputChange}
                  placeholder="Baths"
                />
              </div>
            </div>
            <div className="form-group half">
              <label>Property Condition</label>
              <select 
                name="condition" 
                value={propertyDetails.condition} 
                onChange={handleInputChange}
              >
                <option>Excellent</option>
                <option>Good</option>
                <option>Fair</option>
                <option>Poor</option>
              </select>
              <div className="rating-stars">
                <span className="star filled"></span>
                <span className="star filled"></span>
                <span className="star filled"></span>
                <span className="star filled"></span>
                <span className="star empty"></span>
              </div>
            </div>
          </div>
          
          <div className="form-actions">
            <button className="btn-secondary">Save Draft</button>
            <button className="btn-primary" onClick={analyzeRisk}>Analyze Risk</button>
          </div>
        </div>
        
        <div className="risk-analysis-section">
          <h3>Risk Analysis</h3>
          
          <RiskMeter score={riskScore} />
          
          <RiskFactors factors={riskFactors} />
          
          <div className="ai-recommendation">
            <h4>AI Recommendation</h4>
            <p>This property shows medium risk primarily due to market volatility and rising interest rates in the area. Stable neighborhood growth partially offsets these concerns.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PropertyAnalysis;
