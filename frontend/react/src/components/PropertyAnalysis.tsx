import React, { useState, useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import RiskMeter from './RiskMeter';
import RiskFactors from './RiskFactors';

// Import the AiRecommendation component
import AiRecommendation from './AiRecommendation'; 

const propertyTypeMap: Record<string, string> = {
  'Single-Family Home': 'SingleFamily',
  Condo: 'Condo',
  Townhouse: 'Townhouse',
  'Multi-Family': 'MultiFamily',
};

const conditionMap: Record<string, string> = {
  Excellent: 'Excellent',
  Good: 'Good',
  Fair: 'Fair',
  Poor: 'Poor',
};

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
  const { user, isLoaded, isSignedIn } = useUser();

  const [propertyDetails, setPropertyDetails] = useState<PropertyDetails>({
    location: '123 Main St, Austin, TX 78701',
    propertyType: 'Single-Family Home',
    price: 475000,
    squareFootage: 2350,
    bedrooms: 4,
    bathrooms: 3,
    yearBuilt: 2005,
    condition: 'Good',
  });

  const [riskScore, setRiskScore] = useState<number | null>(null);
  const [aiRecommendation, setAiRecommendation] = useState<string>('');
  const [riskFactors, setRiskFactors] = useState<{ name: string; score: number }[]>([]);

  useEffect(() => {
    if (isLoaded && isSignedIn && user) {
      const initializeUser = async () => {
        try {
          const res = await fetch('http://localhost:5001/user_init', {
            method: 'POST',
            mode: 'cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: user.id }),
          });
          const data = await res.json();
          console.log('User init response:', data);
        } catch (err) {
          console.error('Failed to initialize user:', err);
        }
      };
      initializeUser();
    }
  }, [isLoaded, isSignedIn, user]);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    setPropertyDetails((prev) => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value,
    }));
  };

  const analyzeRisk = async () => {
    if (!isLoaded || !isSignedIn || !user) {
      alert('User data not available. Please sign in.');
      return;
    }
    try {
      const mappedPropertyType = propertyTypeMap[propertyDetails.propertyType] || 'SingleFamily';
      const mappedCondition = conditionMap[propertyDetails.condition] || 'Good';

      const payload = {
        user_id: user.id,
        property: {
          YrSold: 2023,
          SqFt: propertyDetails.squareFootage,
          Bedrooms: propertyDetails.bedrooms,
          Bathrooms: propertyDetails.bathrooms,
          YearBuilt: propertyDetails.yearBuilt,
          Condition: mappedCondition,
          PropertyType: mappedPropertyType,
          Neighborhood: 'NAmes',
        },
      };

      console.log('Sending payload to /analyze:', payload);

      const res = await fetch('http://127.0.0.1:5001/analyze', {
        method: 'POST',
        mode: 'cors',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      console.log('Response from /analyze:', data);

      setRiskScore(data.risk_score ?? null);
      setAiRecommendation(data.ai_recommendation ?? '');
      setRiskFactors([]); // Optionally parse additional risk factors if available.
    } catch (err) {
      console.error('Error analyzing risk:', err);
      alert('Failed to analyze risk. Check console for details.');
    }
  };

  return (
    <div className="property-analysis-container">
      <h2>Property Risk Analysis</h2>
      <p>Input property details to generate a comprehensive risk assessment.</p>
      
      <div className="analysis-content">
        {/* Left Panel: Property Details Input */}
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
                type="number"
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
                  type="number"
                  name="price"
                  value={propertyDetails.price}
                  onChange={handleInputChange}
                />
              </div>
            </div>
            <div className="form-group half">
              <label>Year Built</label>
              <input
                type="number"
                name="yearBuilt"
                value={propertyDetails.yearBuilt}
                onChange={handleInputChange}
              />
            </div>
          </div>
          
          <div className="form-row">
            <div className="form-group half">
              <label>Bedrooms &amp; Bathrooms</label>
              <div className="beds-baths-inputs">
                <input
                  type="number"
                  name="bedrooms"
                  value={propertyDetails.bedrooms}
                  onChange={handleInputChange}
                  placeholder="Beds"
                />
                <input
                  type="number"
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
        
        {/* Right Panel: Risk Analysis Results */}
        <div className="risk-analysis-section">
          <h3>Risk Analysis</h3>
          {riskScore !== null ? (
            <RiskMeter score={riskScore} />
          ) : (
            <p>Risk score not yet available. Click "Analyze Risk" above.</p>
          )}
          <RiskFactors factors={riskFactors} />

          <div className="ai-recommendation">
            <h4>Comprehensive AI Analysis (Pros &amp; Cons)</h4>
            {aiRecommendation ? (
              // Render the AiRecommendation component instead of raw text
              <AiRecommendation content={aiRecommendation} />
            ) : (
              <p>No AI analysis yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PropertyAnalysis;