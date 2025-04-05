//Dashboard.tsx

import React, { useEffect } from "react";

const Dashboard: React.FC = () => {
  useEffect(() => {
    fetch("http://localhost:5000/api/transactions")
      .then((res) => res.json())
      .then((data) => {
        console.log("Fetched data:", data);
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-blue-600 p-4 text-white">
        <h1 className="text-xl font-bold">Financial Time Machine</h1>
      </nav>
      <main className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Past Section */}
          <section className="bg-white rounded shadow p-4">
            <h2 className="text-lg font-semibold mb-2">Past</h2>
            <p className="text-gray-600">Placeholder for historical spend review.</p>
          </section>
          {/* Present Section */}
          <section className="bg-white rounded shadow p-4">
            <h2 className="text-lg font-semibold mb-2">Present</h2>
            <p className="text-gray-600">Placeholder for real-time budget analysis.</p>
          </section>
          {/* Future Section */}
          <section className="bg-white rounded shadow p-4">
            <h2 className="text-lg font-semibold mb-2">Future</h2>
            <p className="text-gray-600">Placeholder for predictive forecasting.</p>
          </section>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;