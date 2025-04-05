// frontend/react/src/components/RiskFactors.tsx
import React from 'react';

interface RiskFactor {
  name: string;
  score: number;
}

interface RiskFactorsProps {
  factors: RiskFactor[];
}

const RiskFactors: React.FC<RiskFactorsProps> = ({ factors }) => {
  const getRiskLevel = (score: number) => {
    if (score > 70) return 'high';
    if (score > 50) return 'medium';
    return 'low';
  };

  return (
    <div className="risk-factors">
      <h4>Risk Factors</h4>
      {factors.map((factor, index) => {
        const riskLevel = getRiskLevel(factor.score);
        return (
          <div className="risk-factor-item" key={index}>
            <div className="factor-name">{factor.name}</div>
            <div className="factor-bar-container">
              <div 
                className={`factor-bar ${riskLevel}`}
                style={{ width: `${factor.score}%` }}
              ></div>
            </div>
            <div className="factor-score">{factor.score}%</div>
          </div>
        );
      })}
    </div>
  );
};

export default RiskFactors;
