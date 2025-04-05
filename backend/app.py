from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/transactions')
def transactions():
    placeholder_data = [
        {"id": 1, "description": "Test Transaction 1", "amount": 100},
        {"id": 2, "description": "Test Transaction 2", "amount": 200},
    ]
    return jsonify(placeholder_data)

if __name__ == '__main__':
    app.run(debug=True)
