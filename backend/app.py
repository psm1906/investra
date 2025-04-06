import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load Nessie API key from environment (required for all Nessie API calls)
NESSIE_API_KEY = os.getenv("NESSIE_API_KEY")
if not NESSIE_API_KEY:
    raise RuntimeError("Missing Nessie API key. Set NESSIE_API_KEY as an env variable.")

# Base URL for Nessie API (Nessie sandbox)
BASE_URL = "http://api.nessieisreal.com"

# Persistent store for Clerk user ID -> Nessie customer ID mapping.
# Using a JSON file for simplicity; replace with a database if needed.
MAPPING_FILE = "user_nessie_map.json"
try:
    with open(MAPPING_FILE, "r") as f:
        user_map = json.load(f)  # dict: {clerk_user_id: nessie_customer_id}
except FileNotFoundError:
    user_map = {}

def save_user_map():
    """Persist the user_map dictionary to the JSON file."""
    with open(MAPPING_FILE, "w") as f:
        json.dump(user_map, f)

def create_nessie_customer():
    """Create a new fake customer in the Nessie sandbox and return the customer ID."""
    url = f"{BASE_URL}/customers?key={NESSIE_API_KEY}"
    customer_payload = {
        "first_name": "John",
        "last_name": "Doe",
        "address": {
            "street_number": "123",
            "street_name": "Main Street",
            "city": "Sampletown",
            "state": "CA",
            "zip": "90401"
        }
    }
    response = requests.post(url, json=customer_payload)
    if response.status_code in (200, 201):
        data = response.json()
        if isinstance(data, dict):
            if data.get("objectCreated"):
                return data["objectCreated"].get("_id") or data["objectCreated"].get("id")
            return data.get("_id") or data.get("id")
    app.logger.error(f"Nessie customer creation failed: {response.status_code}, {response.text}")
    return None

def create_nessie_account(customer_id, account_type, balance=0):
    """Create an account (Checking or Savings) for the given customer_id. Returns the account ID."""
    url = f"{BASE_URL}/customers/{customer_id}/accounts?key={NESSIE_API_KEY}"
    account_payload = {
        "type": account_type,               # "Checking" or "Savings"
        "nickname": f"{account_type} Account",
        "rewards": 0,
        "balance": balance
    }
    response = requests.post(url, json=account_payload)
    if response.status_code in (200, 201):
        data = response.json()
        if data.get("objectCreated"):
            return data["objectCreated"].get("_id") or data["objectCreated"].get("id")
        return data.get("_id") or data.get("id")
    app.logger.error(f"Failed to create {account_type} account: {response.status_code}, {response.text}")
    return None

def create_nessie_deposit(account_id, amount, description):
    """Create a deposit transaction for the given account (adds funds)."""
    url = f"{BASE_URL}/accounts/{account_id}/deposits?key={NESSIE_API_KEY}"
    transaction_payload = {
        "medium": "balance",
        "amount": amount,
        "description": description
    }
    response = requests.post(url, json=transaction_payload)
    if response.status_code not in (200, 201):
         app.logger.error(f"Failed deposit: {response.status_code}, {response.text}")

def create_nessie_withdrawal(account_id, amount, description):
    """Create a withdrawal transaction for the given account (removes funds)."""
    url = f"{BASE_URL}/accounts/{account_id}/withdrawals?key={NESSIE_API_KEY}"
    transaction_payload = {
        "medium": "balance",
        "amount": amount,
        "description": description
    }
    response = requests.post(url, json=transaction_payload)
    if response.status_code not in (200, 201):
         app.logger.error(f"Failed withdrawal: {response.status_code}, {response.text}")

def ensure_nessie_customer(clerk_user_id):
    """
    Ensure the given Clerk user has a corresponding Nessie customer (create if not exists).
    Returns the Nessie customer ID.
    """
    if clerk_user_id in user_map:
        return user_map[clerk_user_id]
    cust_id = create_nessie_customer()
    if not cust_id:
        return None
    # Create Checking and Savings accounts for the new customer
    checking_acct = create_nessie_account(cust_id, "Checking", balance=5000)
    savings_acct  = create_nessie_account(cust_id, "Savings",  balance=10000)
    # Generate a few fake transactions for demo purposes
    if checking_acct:
        create_nessie_deposit(checking_acct, 1000, "Initial deposit")
        create_nessie_withdrawal(checking_acct, 200, "Grocery Shopping")
        create_nessie_withdrawal(checking_acct, 150, "Utility Bill")
    if savings_acct:
        create_nessie_deposit(savings_acct, 500, "Initial deposit")
        create_nessie_deposit(savings_acct, 50,  "Monthly Interest")
    user_map[clerk_user_id] = cust_id
    save_user_map()
    return cust_id

@app.route('/user_init', methods=['POST'])
def user_init():
    """
    Endpoint to initialize user data after login.
    Expects JSON with {"user_id": <Clerk user ID>}.
    Creates Nessie customer/accounts if not already done.
    """
    data = request.get_json()
    clerk_user_id = data.get("user_id")
    if not clerk_user_id:
        return jsonify({"error": "Missing user_id"}), 400
    nessie_id = ensure_nessie_customer(clerk_user_id)
    if not nessie_id:
        return jsonify({"error": "Failed to initialize Nessie data"}), 500
    return jsonify({"message": "Nessie customer ready", "customer_id": nessie_id})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analysis endpoint that uses the user's Nessie data.
    Expects JSON with {"user_id": <Clerk user ID>, ...}.
    """
    data = request.get_json()
    clerk_user_id = data.get("user_id")
    if not clerk_user_id:
        return jsonify({"error": "Missing user_id"}), 400
    # Lookup Nessie customer ID (create if necessary)
    nessie_id = user_map.get(clerk_user_id)
    if not nessie_id:
        nessie_id = ensure_nessie_customer(clerk_user_id)
        if not nessie_id:
            return jsonify({"error": "No Nessie data for user"}), 500
    # Example analysis: retrieve the user's accounts from Nessie
    acct_url = f"{BASE_URL}/customers/{nessie_id}/accounts?key={NESSIE_API_KEY}"
    accounts_resp = requests.get(acct_url)
    accounts = accounts_resp.json() if accounts_resp.status_code == 200 else {}
    result = {"accounts": accounts}
    return jsonify({"customer_id": nessie_id, "analysis_result": result})

if __name__ == '__main__':
    app.run(debug=True, port=5001)