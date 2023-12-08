# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# List to store layer information
layers = {}

@app.route('/')
def index():
    return render_template('index.html')

# Temp json
data = {
    'item': 'value'
}

@app.route('/edit-network', methods=['GET'])
def edit_network():
    return render_template('index2.html', data=data)

@app.route('/save-network', methods=['POST'])
def save_network():
    # Get data from request
    data = request.get_json()
    print(data)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
