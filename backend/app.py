from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests

load_dotenv()

app = Flask(__name__)

# Load API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NESSIE_API_KEY = os.getenv("NESSIE_API_KEY") 

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Helper: Build Prompt for Gemini
def build_prompt(property_data, financial_data, user_goals):
    return f"""
    A user is evaluating a real estate investment.

    🏠 Property:
    - Type: {property_data.get('type', 'N/A')}
    - Zip Code: {property_data.get('zip', 'N/A')}
    - Price: ${property_data.get('price', 'N/A')}

    💰 Banking Summary (via Capital One):
    - Accounts: {financial_data.get('accounts', 'N/A')}
    - Avg. Balance: ${financial_data.get('average_balance', 'N/A')}

    🎯 Goals: {user_goals or 'Not specified'}

    Analyze the investment risk (Low, Medium, High), explain in plain English, and give recommendations.
    """

# Capital One Banking Data (Mocked)
def fetch_mock_banking_data():
    try:
        # Replace with actual Nessie endpoints you're using
        url = f"http://api.reimaginebanking.com/accounts?key={NESSIE_API_KEY}"
        response = requests.get(url)
        data = response.json()

        # Example: Aggregate or summarize mock data
        avg_balance = sum(acc.get('balance', 0) for acc in data) / len(data)
        return {
            "accounts": len(data),
            "average_balance": round(avg_balance, 2)
        }
    except Exception as e:
        print("Nessie error:", e)
        return {
            "accounts": 0,
            "average_balance": 0
        }

@app.route('/', methods=['GET'])
def index():
    return "✅ Real Estate Risk Analyzer with Gemini + Capital One is running!"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    # Extract data
    property_data = data.get("property", {})
    user_goals = data.get("user_goals", "")

    # Get banking summary from Nessie
    financial_data = fetch_mock_banking_data()

    # Build prompt
    prompt = build_prompt(property_data, financial_data, user_goals)

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return jsonify({
            "model_used": "gemini",
            "risk_analysis": response.text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
