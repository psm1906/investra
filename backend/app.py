# backend/app.py
from flask import Flask, jsonify, request
from properties_routes import properties_bp
from rag_model import analyze_investment_risk_api

app = Flask(__name__)

# Register the blueprint for /api/properties
app.register_blueprint(properties_bp, url_prefix='/api')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint for AI-based risk analysis using LightGBM + Gemini + Nessie data.
    Expects JSON like:
    {
      "user_id": "demoUser123",
      "property": { ...fields... }
    }
    """
    payload = request.get_json()
    user_id = payload.get("user_id", "default-user-id") 
    user_input = payload.get("property", {})
    try:
        result = analyze_investment_risk_api(user_id, user_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local development:
    app.run(debug=True, port=5000)