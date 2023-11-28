# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# List to store layer information
layers = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/settings')
def settings():
    # Add your settings logic here
    return render_template('settings.html')

@app.route('/save_layer', methods=['POST'])
def save_layer():
    data = request.get_json()
    layer_id = data.get('layerId')
    input_size = data.get('inputSize')
    output_size = data.get('outputSize')

    if layer_id is None:
        return jsonify({'code': '400', 'message': 'Layer ID cannot be empty'})
    if input_size is None:
        return jsonify({'code': '400', 'message': 'Input size cannot be empty'})
    if output_size is None:
        return jsonify({'code': '400', 'message': 'Output size cannot be empty'})

    # TODO: Type checks for layer_id, input_size and output_size
    input_size = int(input_size)
    output_size = int(output_size)

    layers[layer_id] = {
        'inputSize': input_size,
        'outputSize': output_size
    }

    # TODO: Checks for valid layer structure
    if input_size == 0:
        return jsonify({'code': '400', 'message': 'Matrix size cannot be zero'})
    if output_size == 0:
        return jsonify({'code': '400', 'message': 'Matrix size cannot be zero'})

    return jsonify({'code': '200', 'message': 'Layer saved'})

if __name__ == '__main__':
    app.run(debug=True)
