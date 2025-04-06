#rag_model.py
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

import datetime
import pandas as pd
from model import predict_risk_score_from_ui

############################################
# 1. Load environment variables & configure
############################################
load_dotenv()
NESSIE_API_KEY = os.getenv("NESSIE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

############################################
# 2. Functions to fetch Nessie data & build user summary
############################################
def fetch_and_summarize_nessie(user_id: str) -> str:
    """
    Fetch user’s financial data from Capital One's Nessie sandbox API.
    Return a short text summary describing the user’s finances:
    - Checking/savings balances
    - Recent big transactions
    - Any credit line usage
    """
    # Example endpoints (replace with your actual usage):
    #   GET /customers/{customer_id}
    #   GET /accounts?key=NESSIE_API_KEY
    #   GET /accounts/{account_id}/transactions
    # Note: user_id might map to Nessie’s internal customer_id or account_id

    try:
        # Example: fetch all accounts for the user
        base_url = f"http://api.reimaginebanking.com"
        resp = requests.get(
            f"{base_url}/customers/{user_id}?key={NESSIE_API_KEY}"
        )
        if resp.status_code != 200:
            return "Could not fetch Nessie data."

        customer_json = resp.json()
        # Suppose we want balances or last transactions:
        # This is just pseudo-code; adapt to match the Nessie response structure:
        accounts_resp = requests.get(
            f"{base_url}/customers/{user_id}/accounts?key={NESSIE_API_KEY}"
        )
        accounts = accounts_resp.json()
        # Build a small text summary:
        total_balance = 0
        for acct in accounts:
            total_balance += acct.get("balance", 0)

        # For real usage, you’d parse transactions, credit usage, etc.
        summary = (
            f"This user’s total bank balance is approximately ${total_balance:.2f}. "
            "They have the following accounts:\n"
        )
        for acct in accounts:
            summary += f" - {acct.get('type')} with balance ${acct.get('balance', 0)}\n"

        return summary.strip()

    except Exception as e:
        return f"Error retrieving Nessie data: {str(e)}"


############################################
# 3. Minimal “retrieval” step
############################################
def retrieve_docs(context_query: str) -> str:
    """
    A placeholder for a real RAG retrieval system.
    In a production app, you’d embed your knowledge-base docs and do similarity search.
    For now, we’ll just return some static text if we detect certain keywords.
    """

    # You might store real documents like "interest_rates.txt", "credit_usage.txt", etc.
    # For a simple example, we just do keyword-based snippets:
    # (In a real system, you’d do embedding-based retrieval.)

    docs_snippets = []

    if "interest" in context_query.lower():
        docs_snippets.append(
            "High interest rates can reduce home affordability, impacting demand and possibly lowering property values."
        )
    if "balance" in context_query.lower():
        docs_snippets.append(
            "Having healthy cash reserves can mitigate risk, since it covers mortgage payments during downturns."
        )
    if "neighborhood" in context_query.lower():
        docs_snippets.append(
            "Neighborhood growth is a positive sign for property appreciation, but also can raise property taxes."
        )

    # Return them as a single string or a list. For prompting, we often just inline them:
    return "\n".join(docs_snippets)


############################################
# 4. Generate an LLM-based explanation that merges risk, Nessie data, & doc retrieval
############################################
def generate_rag_analysis(property_data: str, user_finance_summary: str) -> str:
    """
    Prompts Gemini with a combination of:
      - property_data (including risk score)
      - user_finance_summary (from Nessie)
      - retrieved doc snippets
    and requests a short “Pros, Cons, and suggested improvements” style explanation.
    """

    # Some attempt at a “context query” to feed retrieval:
    # We might guess from the property data that interest rates are relevant, or we might
    # just send the entire text. This is flexible:
    context_query = (property_data + "\n" + user_finance_summary).lower()

    # Retrieve doc snippets:
    relevant_snippets = retrieve_docs(context_query)

    # Build a prompt that references all these pieces:
    prompt = f"""
You are a helpful real estate investment assistant. A user wants a property risk analysis.

PROPERTY & RISK DETAILS:
{property_data}

USER FINANCIAL SUMMARY:
{user_finance_summary}

RELEVANT BACKGROUND DOCUMENTS:
{relevant_snippets}

TASK: 
1) Summarize the overall risk level (0-100 scale).
2) Provide "Pros" and "Cons" based on both the property and the user's financial position.
3) Suggest improvements or next steps the user could take to reduce risk or strengthen their position.

Respond in a concise, user-friendly tone.
"""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating analysis: {str(e)}"


############################################
# 5. High-level function to get final JSON for the UI
############################################
def analyze_investment_risk_api(user_id: str, user_input: dict):
    """
    Merges property-based risk (LightGBM) and user finance data (Nessie),
    then calls the LLM for a combined explanation.

    Returns a JSON-style dict suitable for a Flask or FastAPI response.
    """
    # 1) Get numeric risk score from model
    risk_score = predict_risk_score_from_ui(user_input)
    # property_data string for the prompt
    property_summary = (
        f"Property info: {user_input}\n"
        f"Model-predicted risk score: {risk_score:.2f} (0-100 scale)."
    )

    # 2) Fetch & summarize Nessie data
    user_finance_summary = fetch_and_summarize_nessie(user_id)

    # 3) Generate the final explanation from the LLM
    rag_explanation = generate_rag_analysis(property_summary, user_finance_summary)

    # 4) Optionally parse out sub-parts of the text if you want a separate “Pros/Cons”
    # but for simplicity, we just return the entire text.
    generated_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return {
        "risk_score": risk_score,
        "ai_recommendation": rag_explanation,
        "property_data": user_input,
        "user_finance_summary": user_finance_summary,
        "generated_date": generated_date
    }