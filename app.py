# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# List to store layer information
layers = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_layer', methods=['POST'])
def save_layer():
    data = request.get_json()
    layer_id = data.get('layerId')
    input_size = data.get('inputSize')
    output_size = data.get('outputSize')

    layers[layer_id] = {
        'inputSize': input_size,
        'outputSize': output_size
    }

    print(layers)

    # For demonstration purposes, returning the saved layers
    return jsonify({'code': '201'})

if __name__ == '__main__':
    app.run(debug=True)
