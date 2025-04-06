# backend/rag_model.py

import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import datetime

# Import the LightGBM model prediction function from model.py
from model import predict_risk_score_from_ui

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API using the GEMINI_API_KEY from .env
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # Ensure GEMINI_API_KEY is set in your .env file

def fetch_banking_data():
    """
    Fetch financial data from the Nessie API.
    """
    nessie_api_key = os.getenv("NESSIE_API_KEY")  # Your Nessie API Key
    url = f"http://api.reimaginebanking.com/accounts?key={nessie_api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Return the JSON data if successful
    else:
        return {"error": "Failed to fetch data from Nessie API"}

def generate_risk_analysis(property_data, financial_data):
    """
    Generate risk analysis using the Gemini API.
    The prompt includes both the property details (with the predicted risk score)
    and the financial summary.
    """
    prompt = f"""
    Property Details: {property_data}
    Financial Summary: {financial_data}
    
    Based on the property details and current financial data, provide a comprehensive analysis of the investment risk. 
    Highlight key factors contributing to the risk score and suggest ways to mitigate these risks.
    """
    
    try:
        # Initialize and call the Gemini model
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating analysis: {str(e)}"

def analyze_investment_risk(user_input):
    """
    Analyze investment risk using the LightGBM model and Gemini API.
    Returns a tuple: (risk_score, risk_analysis, property_details)
    """
    # Predict risk score using the LightGBM model
    risk_score = predict_risk_score_from_ui(user_input)
    
    # Format the property details including the predicted risk score
    property_details = f"User Input: {user_input}\nPredicted Risk Score: {risk_score:.2f}"
    
    # Fetch financial data from the Nessie API
    financial_data = fetch_banking_data()
    if 'error' in financial_data:
        raise ValueError(financial_data['error'])
    
    # Generate risk analysis using the Gemini API (RAG-based)
    risk_analysis = generate_risk_analysis(property_details, financial_data)
    
    return risk_score, risk_analysis, property_details

def analyze_investment_risk_api(user_input):
    """
    API-compatible function to analyze investment risk.
    
    This function returns a JSON-serializable dictionary containing:
      - risk_score: The numerical risk score (0-100)
      - risk_analysis: A textual explanation of the risk
      - property_details: Details of the property and predicted risk
      - report_title: A title for the report
      - generated_date: The date the report was generated
    
    This output can be consumed by Flask endpoints and then sent to the frontend.
    """
    try:
        risk_score, risk_analysis, property_details = analyze_investment_risk(user_input)
        # Use current date for generated_date
        generated_date = datetime.datetime.now().strftime("%Y-%m-%d")
        report_title = "Market Risk Analysis"
        
        return {
            "risk_score": risk_score,
            "risk_analysis": risk_analysis,
            "property_details": property_details,
            "report_title": report_title,
            "generated_date": generated_date
        }
    except Exception as e:
        return {"error": str(e)}

# For quick testing purposes, run an example if this script is executed directly.
if __name__ == "__main__":
    # Example test input representing user-provided property details.
    test_input = {
        "YrSold": 2020,
        "SqFt": 2000,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "YearBuilt": 1990,
        "Condition": "Good",
        "Neighborhood": "NAmes",
        "PropertyType": "SingleFamily",
        "MortgageRate": 3.5,
        "UnemploymentRate": 4.0,
        "CPI": 2.0
    }
    report = analyze_investment_risk_api(test_input)
    print("API Report:")
    print(report)