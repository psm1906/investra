import React from 'react';

const Reports: React.FC = () => {
  return (
    <div className="dashboard-container">
      <div className="main-content">
        <div className="page-content">
          <h2>Reports</h2>
          <p>View and download investment reports</p>
          
          <div className="reports-list">
            <div className="report-item">
              <div className="report-icon"></div>
              <div className="report-info">
                <h3>Q1 2025 Investment Summary</h3>
                <p>Generated on April 1, 2025</p>
              </div>
              <button className="download-btn">Download</button>
            </div>
            
            <div className="report-item">
              <div className="report-icon"></div>
              <div className="report-info">
                <h3>Market Risk Analysis</h3>
                <p>Generated on March 15, 2025</p>
              </div>
              <button className="download-btn">Download</button>
            </div>
            
            <div className="report-item">
              <div className="report-icon"></div>
              <div className="report-info">
                <h3>Property Comparison Report</h3>
                <p>Generated on February 28, 2025</p>
              </div>
              <button className="download-btn">Download</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Reports;
