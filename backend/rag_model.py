# rag_model.py
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

import datetime
import pandas as pd
from models.model import predict_risk_score_from_ui

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
    Fetch user’s financial data from the Nessie API.
    Return a summary describing:
    - Customer basic info
    - Checking/savings balances from accounts
    - Recent transaction history for each account
    """
    base_url = "http://api.nessieisreal.com"
    try:
        # Fetch customer details
        cust_resp = requests.get(
            f"{base_url}/customers/{user_id}?key={NESSIE_API_KEY}"
        )
        if cust_resp.status_code != 200:
            return "Could not fetch customer details from Nessie."
        customer = cust_resp.json()

        # Fetch all accounts for the user
        accounts_resp = requests.get(
            f"{base_url}/customers/{user_id}/accounts?key={NESSIE_API_KEY}"
        )
        if accounts_resp.status_code != 200:
            return "Could not fetch Nessie accounts data."
        accounts = accounts_resp.json()

        total_balance = 0
        account_details = ""
        transaction_details = ""
        
        # Loop through each account to accumulate balances and fetch transactions
        for acct in accounts:
            acct_id = acct.get("id")
            balance = acct.get("balance", 0)
            total_balance += balance
            acct_type = acct.get("type", "Unknown")
            account_details += f" - {acct_type} (ID: {acct_id}) with balance ${balance:.2f}\n"
            
            # Fetch transactions for this account
            trans_resp = requests.get(
                f"{base_url}/accounts/{acct_id}/transactions?key={NESSIE_API_KEY}"
            )
            if trans_resp.status_code == 200:
                transactions = trans_resp.json()
                # Use up to the 3 most recent transactions (if available)
                recent = transactions[-3:] if len(transactions) >= 3 else transactions
                if recent:
                    transaction_details += f"Account {acct_id} recent transactions:\n"
                    for t in recent:
                        amount = t.get("amount", "N/A")
                        date = t.get("transaction_date", "N/A")
                        description = t.get("description", "No description")
                        transaction_details += f"   - ${amount} on {date}: {description}\n"
                else:
                    transaction_details += f"Account {acct_id} has no transactions.\n"
            else:
                transaction_details += f"Could not fetch transactions for account {acct_id}.\n"

        # Build a summary string
        summary = (
            f"Customer: {customer.get('first_name', '')} {customer.get('last_name', '')}\n"
            f"Total Bank Balance: ${total_balance:.2f}\n"
            f"Accounts:\n{account_details.strip()}\n\n"
            f"Transaction History:\n{transaction_details.strip()}"
        )
        return summary

    except Exception as e:
        return f"Error retrieving Nessie data: {str(e)}"

############################################
# 3. Minimal “retrieval” step
############################################
def retrieve_docs(context_query: str) -> str:
    """
    A placeholder for a real RAG retrieval system.
    Returns static background documents based on keywords.
    """
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
    return "\n".join(docs_snippets)

############################################
# 4. Generate an LLM-based explanation that merges risk, Nessie data, & doc retrieval
############################################
def generate_rag_analysis(property_data: str, user_finance_summary: str) -> str:
    """
    Prompts Gemini with a combination of:
      - Property data (including risk score)
      - User financial summary from Nessie
      - Retrieved document snippets
    Returns a concise explanation including risk, pros/cons, and suggestions.
    """
    context_query = (property_data + "\n" + user_finance_summary).lower()
    relevant_snippets = retrieve_docs(context_query)
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
    Merges property-based risk (LightGBM) and customer financial history (Nessie),
    then calls the LLM for a combined explanation.
    Returns a JSON-style dict for the UI.
    """
    # 1) Get numeric risk score from model
    risk_score = predict_risk_score_from_ui(user_input)
    property_summary = (
        f"Property info: {user_input}\n"
        f"Model-predicted risk score: {risk_score:.2f} (0-100 scale)."
    )

    # 2) Fetch & summarize Nessie customer history data
    user_finance_summary = fetch_and_summarize_nessie(user_id)

    # 3) Generate the final explanation from the LLM
    rag_explanation = generate_rag_analysis(property_summary, user_finance_summary)

    generated_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return {
        "risk_score": risk_score,
        "ai_recommendation": rag_explanation,
        "property_data": user_input,
        "user_finance_summary": user_finance_summary,
        "generated_date": generated_date
    }