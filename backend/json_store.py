# backend/json_store.py
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)  # ensure data folder

DATA_FILE = os.path.join(DATA_DIR, "properties.json")

def load_properties():
    """Load a list of properties from properties.json."""
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_properties(properties_list):
    """Overwrite properties.json with the updated list of property dicts."""
    with open(DATA_FILE, 'w') as f:
        json.dump(properties_list, f, indent=2)