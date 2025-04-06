import React from 'react';

const Analytics: React.FC = () => {
  return (
    <div className="dashboard-container">
      {/* Main Content */}
      <div className="main-content">
        <div className="page-content">
          <h2>Analytics</h2>
          <p>Gain insights into your investments and risk management strategies.</p>

          {/* Analytics Content */}
          <div className="analytics-charts">
            {/* Investment Distribution Section */}
            <div className="chart-container">
              <h3>Investment Distribution</h3>
              <p>Analyze how your investments are distributed across various categories.</p>
              <img src="/src/assets/images/InvestDist.jpg" alt="Investment Distribution" className="analytics-chart" />
            </div>

            {/* Risk Management Section */}
            <div className="chart-container">
              <h3>Risk Management</h3>
              <p>Understand the risk levels associated with your portfolio.</p>
              <img src="/src/assets/images/riskManage.png" alt="Risk Management" className="analytics-chart" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
