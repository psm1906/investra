from flask import Flask, jsonify, request
from rag_model import analyze_investment_risk

app = Flask(__name__)

# Flask route for analyzing real estate risk
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()  # Get the data sent in the POST request
    
    # Fetch property details from the incoming request
    property_data = data.get("property", {})
    
    # Generate risk analysis using RAG-based model (from rag_model.py)
    try:
        risk_analysis = analyze_investment_risk(property_data)
        return jsonify({"risk_analysis": risk_analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
