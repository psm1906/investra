import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # Ensure GEMINI_API_KEY is in your .env file

# Fetch banking data from Nessie API
def fetch_banking_data():
    nessie_api_key = os.getenv("NESSIE_API_KEY")  # Your Nessie API Key
    url = f"http://api.reimaginebanking.com/accounts?key={nessie_api_key}"
    
    response = requests.get(url)
    
    # If the request is successful
    if response.status_code == 200:
        return response.json()  # Return the JSON data
    else:
        return {"error": "Failed to fetch data from Nessie API"}

# Generate risk analysis using Gemini API (RAG Model)
def generate_risk_analysis(property_data, financial_data):
    # Format your prompt with the necessary information
    prompt = f"""
    Property Details: {property_data}
    Financial Summary: {financial_data}
    
    Based on this information, provide an analysis of the investment risk, including any potential risks for the investor.
    """
    
    try:
        # Call Gemini API for generating content based on the prompt
        model = genai.GenerativeModel('gemini-pro')  # Specify the model to use
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating analysis: {str(e)}"

# Main function to analyze investment risk using RAG model
def analyze_investment_risk(property_data):
    # Step 1: Fetch financial data using Nessie API
    financial_data = fetch_banking_data()
    
    # If there was an error in fetching the financial data
    if 'error' in financial_data:
        raise ValueError(financial_data['error'])
    
    # Step 2: Generate risk analysis using Gemini (RAG-based model)
    risk_analysis = generate_risk_analysis(property_data, financial_data)
    
    # Return the generated risk analysis
    return risk_analysis
