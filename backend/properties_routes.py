# backend/properties_routes.py
from flask import Blueprint, request, jsonify
from json_store import load_properties, save_properties

properties_bp = Blueprint('properties', __name__)

@properties_bp.route('/properties', methods=['GET'])
def list_properties():
    """
    Returns all properties in the JSON file.
    In a real multi-user setup, you'd filter by user ID 
    or verify token & parse user ID from there.
    """
    props = load_properties()
    return jsonify(props), 200

@properties_bp.route('/properties', methods=['POST'])
def add_property():
    """
    Add a new property to the JSON file.
    JSON body example:
    {
      "address": "789 Pine Rd, Houston, TX",
      "propertyType": "Single-Family Home",
      "price": 420000,
      "riskScore": 72
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    props = load_properties()
    
    next_id = max([p["id"] for p in props], default=0) + 1
    
    new_prop = {
        "id": next_id,
        "address": data.get("address", "Unknown"),
        "propertyType": data.get("propertyType", "Single-Family"),
        "price": data.get("price", 0),
        "riskScore": data.get("riskScore", 0)
        # add other fields if needed
    }
    props.append(new_prop)
    
    save_properties(props)
    return jsonify(new_prop), 201

@properties_bp.route('/properties/<int:prop_id>', methods=['DELETE'])
def delete_property(prop_id):
    """
    Delete a property by ID from the JSON file.
    """
    props = load_properties()
    updated = [p for p in props if p["id"] != prop_id]
    if len(updated) == len(props):
        # No property removed => not found
        return jsonify({"error": "Property not found"}), 404
    
    save_properties(updated)
    return jsonify({"message": f"Property {prop_id} deleted"}), 200
