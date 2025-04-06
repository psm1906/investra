import React, { useState } from 'react';
import RiskMeter from './RiskMeter';
import RiskFactors from './RiskFactors';

/** 
 * Map from UI dropdown to the strings the model expects in `predict_risk_score_from_ui()`.
 */
const propertyTypeMap: Record<string, string> = {
  'Single-Family Home': 'SingleFamily',
  Condo: 'Condo',
  Townhouse: 'Townhouse',
  'Multi-Family': 'MultiFamily', // model doesn't have MultiFamily specifically, but we'll pass "MultiFamily"
};

const conditionMap: Record<string, string> = {
  Excellent: 'Excellent',
  Good: 'Good',
  Fair: 'Fair',
  Poor: 'Poor',
};

interface PropertyDetails {
  location: string;
  propertyType: string; // e.g. "Single-Family Home"
  price: number;
  squareFootage: number;
  bedrooms: number;
  bathrooms: number;
  yearBuilt: number;
  condition: string;    // e.g. "Good"
}

const PropertyAnalysis: React.FC = () => {
  // Initial form state
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

  // After the backend call, store the returned data:
  const [riskScore, setRiskScore] = useState<number | null>(null);
  const [aiRecommendation, setAiRecommendation] = useState<string>('');
  const [riskFactors, setRiskFactors] = useState<{ name: string; score: number }[]>([]);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setPropertyDetails((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Call the backend to analyze risk
  const analyzeRisk = async () => {
    try {
      // Suppose we track the user ID in your app; hardcode or get from context/auth:
      const userId = 'demoUser123';

      // Construct the payload: map UI fields -> what the model expects
      const mappedPropertyType = propertyTypeMap[propertyDetails.propertyType] || 'SingleFamily';
      const mappedCondition = conditionMap[propertyDetails.condition] || 'Good';

      const payload = {
        user_id: userId,
        property: {
          // The model expects "YrSold" (year sold) - you can pick 2023 or any year
          YrSold: 2023,

          // The model expects "SqFt" -> "GrLivArea"
          SqFt: propertyDetails.squareFootage,

          // The model expects "Bedrooms" -> "BedroomAbvGr"
          Bedrooms: propertyDetails.bedrooms,

          // The model expects "Bathrooms" -> "FullBath"
          Bathrooms: propertyDetails.bathrooms,

          // The model expects "YearBuilt"
          YearBuilt: propertyDetails.yearBuilt,

          // The model expects "Condition" to be one of "Excellent","Good","Fair","Poor"
          Condition: mappedCondition,

          // The model expects "PropertyType" -> "SingleFamily","Townhouse","Condo", etc.
          PropertyType: mappedPropertyType,

          // The model also references "Neighborhood", default is "NAmes"
          Neighborhood: 'NAmes',

          // If you'd like to pass MortgageRate, etc., you could do:
          // MortgageRate: 6.5,

          // Price is not used by the model, but we keep it for UI consistency:
          // Price: propertyDetails.price,
        },
      };

      console.log('Sending payload to /analyze:', payload);

      const res = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      console.log('Response from /analyze:', data);

      // The backend returns e.g.:
      // {
      //   risk_score: 78.23,
      //   ai_recommendation: "LLM text...",
      //   property_data: {...},
      //   user_finance_summary: "...",
      //   generated_date: "2025-04-06"
      // }
      setRiskScore(data.risk_score ?? null);
      setAiRecommendation(data.ai_recommendation ?? '');
      setRiskFactors([]); // Or parse data.risk_factors if provided
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
          
          {/* Location */}
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
          
          {/* Property Image Upload (Placeholder) */}
          <div className="property-image-section">
            <div className="image-placeholder">
              <div className="house-image"></div>
            </div>
            <div className="image-upload-info">
              <h4>Property Images</h4>
              <p>Upload photos or drag and drop files here</p>
            </div>
          </div>
          
          {/* Property Type & SqFt */}
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
          
          {/* Price & Year Built */}
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
          
          {/* Bedrooms & Bathrooms & Condition */}
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
          
          {/* If riskScore is null, user hasn’t run analysis yet or no result */}
          {riskScore !== null ? (
            <RiskMeter score={riskScore} />
          ) : (
            <p>Risk score not yet available. Click "Analyze Risk" above.</p>
          )}

          <RiskFactors factors={riskFactors} />

          <div className="ai-recommendation">
            <h4>AI Recommendation</h4>
            {aiRecommendation ? (
              <p>{aiRecommendation}</p>
            ) : (
              <p>No AI recommendation yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PropertyAnalysis;