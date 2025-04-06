import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import uuid  # For generating a unique identifier that can serve as a name

load_dotenv()
app = Flask(__name__)

# Your existing CORS config
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    expose_headers="Content-Disposition",
    allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials", "Access-Control-Allow-Origin"],
    supports_credentials=True
)

# Make sure this env var is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY as an env variable.")

genai.configure(api_key=GEMINI_API_KEY)

# Get the Gemini Pro model
gemini_pro_model = genai.GenerativeModel('gemini-2.0-flash')

MAPPING_FILE = "user_gemini_map.json"
try:
    with open(MAPPING_FILE, "r") as f:
        user_map = json.load(f)
except FileNotFoundError:
    user_map = {}

def save_user_map():
    with open(MAPPING_FILE, "w") as f:
        json.dump(user_map, f)

def generate_gemini_customer(user_name: str) -> dict:
    """
    Generate a realistic financial profile JSON using a valid Google Generative AI model.
    """
    prompt = f"""
You are a creative assistant tasked with generating a realistic financial profile for a user named "{user_name}".
Generate a valid JSON object with the following structure. Ensure the output is ONLY valid JSON and includes realistic sample data:

{{
  "first_name": "...",
  "last_name": "...",
  "accounts": [
    {{
      "account_id": "...",
      "type": "Checking" or "Savings",
      "nickname": "...",
      "balance": ...,
      "transactions": [
        {{
          "amount": ...,
          "date": "YYYY-MM-DD",
          "description": "..."
        }},
        {{
          "amount": ...,
          "date": "YYYY-MM-DD",
          "description": "..."
        }}
        // ... more transactions
      ]
    }},
    {{
      "account_id": "...",
      "type": "Checking" or "Savings",
      "nickname": "...",
      "balance": ...,
      "transactions": [...]
    }}
    // ... more accounts
  ]
}}

Be creative and generate realistic-looking data. Do NOT include any markdown formatting like ```json around the JSON output.
"""

    try:
        response = gemini_pro_model.generate_content(prompt)

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            app.logger.error(f"Gemini Pro blocked the prompt for user '{user_name}': {response.prompt_feedback.block_reason}")
            return None
        elif response.candidates and not response.candidates[0].content.parts:
            app.logger.error(f"Gemini Pro returned an empty response for user '{user_name}'.")
            return None

        generated_text = response.candidates[0].content.parts[0].text

        # Remove markdown code block if present
        if generated_text.startswith("```json") and generated_text.endswith("```"):
            generated_text = generated_text[7:-3].strip()
        elif generated_text.startswith("```") and generated_text.endswith("```"):
            generated_text = generated_text[3:-3].strip()

        try:
            customer_data = json.loads(generated_text)
            return customer_data
        except json.JSONDecodeError as e:
            app.logger.error(f"Failed to parse generated JSON for user '{user_name}': {e}, Response text: '{generated_text}'")
            return None

    except Exception as e:
        app.logger.error(f"Error generating customer data for user '{user_name}': {e}")
        return None

def ensure_customer_data(clerk_user_id: str, user_name: str) -> dict:
    if clerk_user_id in user_map:
        return user_map[clerk_user_id]

    # Generate new if not present
    customer_data = generate_gemini_customer(user_name)
    if customer_data:
        user_map[clerk_user_id] = customer_data
        save_user_map()
        return customer_data
    else:
        return None

@app.route('/user_init', methods=['POST'])
def user_init():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    clerk_user_id = data.get("user_id")
    user_name = data.get("user_name")
    if not clerk_user_id:
        return jsonify({"error": "Missing user_id"}), 400

    if not user_name:
        # Generate a unique identifier to use as a placeholder name
        user_name = f"User_{uuid.uuid4().hex[:8]}"
        app.logger.info(f"Generated user name for {clerk_user_id}: {user_name}")

    customer_profile = ensure_customer_data(clerk_user_id, user_name)
    if not customer_profile:
        return jsonify({"error": "Failed to generate customer profile"}), 500

    return jsonify({
        "message": "Customer profile generated",
        "customer_profile": customer_profile
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Similar approach, we do property risk analysis
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        clerk_user_id = data.get("user_id")
        property_details = data.get("property")
        if not clerk_user_id:
            return jsonify({"error": "Missing user_id"}), 400
        if not property_details:
            return jsonify({"error": "Missing property details"}), 400

        # If no profile, generate (we might want to fetch existing here instead)
        if clerk_user_id not in user_map:
            # Generate a unique identifier if no user_name is provided
            user_name_for_new = data.get("user_name", f"User_{uuid.uuid4().hex[:8]}")
            customer_profile = generate_gemini_customer(user_name_for_new)
            if customer_profile:
                user_map[clerk_user_id] = customer_profile
                save_user_map()
            else:
                return jsonify({"error": "Failed to generate customer profile for analysis"}), 500

        from backend.rag_model import analyze_investment_risk_api  # Import the correct function
        analysis_result = analyze_investment_risk_api(clerk_user_id, property_details)
        return jsonify(analysis_result)

    except Exception as e:
        app.logger.error("Error in /analyze endpoint: %s", e)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    # Start on port 5001
    app.run(debug=True, port=5001)