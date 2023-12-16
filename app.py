import os
import json
import threading
from queue import Queue

from flask import Flask, render_template, request, jsonify, Response

from main import Classifier


def save_networks_to_file(networks):
    with open('./inc/networks.json', 'w', encoding='utf-8') as f:
        json.dump(networks, f, indent=4)


def load_networks_from_file():
    if not os.path.exists('./inc/networks.json'):
        return {}
    with open('./inc/networks.json', 'r', encoding='utf-8') as f:
        return json.load(f)


app = Flask(__name__)
queue = Queue()


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


@app.route('/get-networks', methods=['GET'])
def get_networks():
    networks = load_networks_from_file()
    return jsonify(networks)


@app.route('/sse')
def sse():
    def event_stream():
        while True:
            result = queue.get()
            yield f'data: {result}\n\n'

    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/train-network', methods=['POST'])
def train_network():

    data = request.get_json()

    network_id = data.get('networkId')
    network = data.get('network')

    thread = threading.Thread(target=train_network_thread, args=(
        network_id, network, queue))
    thread.start()

    return jsonify(data)


def train_network_thread(network_id, network, queue):
    classifier = Classifier()

    queue.put({'data': 'It worked 1!'})
    print('It worked 1!')

    for layer in network['layers'].values():
        classifier.add_layer(
            layer['layerConfig'])

    classifier.train_network(queue=queue)

    # Get networks from file
    networks = load_networks_from_file()

    # Save network to list
    if network_id not in networks:
        networks[network_id] = {}
    networks[network_id]['network'] = classifier.net.info()
    networks[network_id]['trained'] = True

    # Save network to file
    save_networks_to_file(networks)


if __name__ == '__main__':
    app.run(debug=True)
