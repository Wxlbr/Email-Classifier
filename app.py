import os
import json

from flask import Flask, render_template, request, jsonify


def save_networks_to_file(networks):
    with open('./inc/networks.json', 'w', encoding='utf-8') as f:
        json.dump(networks, f, indent=4)


def load_networks_from_file():
    if not os.path.exists('./inc/networks.json'):
        return {}
    with open('./inc/networks.json', 'r', encoding='utf-8') as f:
        return json.load(f)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/edit-network', methods=['GET'])
def edit_network():

    # Get network Id from request
    network_id = request.args.get('networkId')

    # Get network from list
    networks = load_networks_from_file()

    # Get network from list
    if network_id in networks:
        layers = networks[network_id]['layers']
    else:
        layers = {}

    data = {
        'networkId': network_id,
        'layers': layers
    }

    return render_template('index2.html', data=data)


@app.route('/save-network', methods=['POST'])
def save_network():
    # Get data from request
    data = request.get_json()

    network_id = data.get('networkId')
    layers = data.get('layers')

    valid = all(layer['valid'] for layer in layers.values())

    # Get networks from file
    networks = load_networks_from_file()

    # Save network to list
    if network_id not in networks:
        networks[network_id] = {}
    networks[network_id]['layers'] = layers
    networks[network_id]['valid'] = valid

    # Save network to file
    save_networks_to_file(networks)

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
