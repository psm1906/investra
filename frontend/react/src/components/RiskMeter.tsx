// frontend/react/src/components/RiskMeter.tsx
import React from 'react';

interface RiskMeterProps {
  score: number;
}

const RiskMeter: React.FC<RiskMeterProps> = ({ score }) => {
  let riskLevel = 'LOW';
  let riskColor = 'low';
  
  if (score > 70) {
    riskLevel = 'HIGH';
    riskColor = 'high';
  } else if (score > 40) {
    riskLevel = 'MEDIUM';
    riskColor = 'medium';
  }
  
  return (
    <div className="risk-meter">
      <div className="score-display">
        <div className="score-number">{score}</div>
        <div className={`risk-level ${riskColor}`}>{riskLevel} RISK</div>
      </div>
      
      <div className="gauge">
        <div className="gauge-background">
          <div className="gauge-section low"></div>
          <div className="gauge-section medium"></div>
          <div className="gauge-section high"></div>
        </div>
        <div 
          className="gauge-needle" 
          style={{ transform: `rotate(${(score / 100) * 180 - 90}deg)` }}
        ></div>
        <div className="gauge-labels">
          <span className="low-label">LOW</span>
          <span className="medium-label">MEDIUM</span>
          <span className="high-label">HIGH</span>
        </div>
      </div>
    </div>
  );
};

export default RiskMeter;
