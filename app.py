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
history_queue = Queue()


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

        if 'inputSize' in networks[network_id]:
            input_size = networks[network_id]['inputSize']
        if 'outputSize' in networks[network_id]:
            output_size = networks[network_id]['outputSize']

    else:
        layers = {}

    data = {
        'networkId': network_id,
        'layers': layers,
        'inputSize': input_size,
        'outputSize': output_size
    }

    return render_template('index2.html', data=data)


@app.route('/save-network', methods=['POST'])
def save_network():
    # Get data from request
    # print(request.data)

    data = request.get_json()

    network_id = data.get('networkId')

    # Get networks from file
    networks = load_networks_from_file()

    # Save network to list
    if network_id not in networks:
        networks[network_id] = data

    # Save network to file
    save_networks_to_file(networks)

    return jsonify(data)


@app.route('/save-layers', methods=['POST'])
def save_layers():
    # Get data from request
    # print(request.data)

    data = request.get_json()

    network_id = data.get('networkId')
    layers = data.get('layers')
    input_size = data.get('inputSize')
    output_size = data.get('outputSize')

    valid = all(layer['valid']
                for layer in layers.values()) if layers else False

    # Get networks from file
    networks = load_networks_from_file()

    # Save network to list
    if network_id not in networks:
        networks[network_id] = {}
    networks[network_id] = {
        'layers': layers,
        'inputSize': input_size,
        'outputSize': output_size,
        'activeCard': False,
        'trained': False,
        'status': 'inactive',
        'valid': valid,
        'network': {}
    }

    # Save network to file
    save_networks_to_file(networks)

    return jsonify(data)


@app.route('/switch-active-network', methods=['POST'])
def switch_active_card():
    data = request.get_json()

    network_id = data.get('newNetworkId')
    active_id = data.get('activeNetworkId')

    # Get networks from file
    networks = load_networks_from_file()

    # TODO: Add validation

    networks[network_id]['activeCard'] = True
    networks[active_id]['activeCard'] = False

    # Save network to file
    save_networks_to_file(networks)

    return jsonify({'status': 'success'})


@app.route('/get-networks', methods=['GET'])
def get_networks():
    networks = load_networks_from_file()
    return jsonify(networks)


@app.route('/sse')
def sse():
    def event_stream():
        while True:
            result = queue.get()
            result = json.dumps(result)
            history_queue.put(result)  # put it back
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

    # queue.put({'data': 'It worked 1!'})
    # print('It worked 1!')

    for layer in network['layers'].values():
        classifier.add_layer(
            layer['layerConfig'])

    classifier.train_network(queue=queue, netId=network_id)

    # Get networks from file
    networks = load_networks_from_file()

    # Save network to list
    if network_id not in networks:
        networks[network_id] = {}
    networks[network_id]['network'] = classifier.net.info()
    networks[network_id]['trained'] = True

    # Save network to file
    save_networks_to_file(networks)


@app.route('/active-training-sse', methods=['GET'])
def active_sses():
    if history_queue.empty():
        return jsonify({'active': False, 'networkId': None, 'data': None})
    data = json.loads(history_queue.get())
    data["active"] = True
    print('Data: ', data)
    return jsonify(data)


@app.route('/delete-network', methods=['POST'])
def delete_network():
    data = request.get_json()

    network_id = data.get('networkId')

    # Get networks from file
    networks = load_networks_from_file()

    print('Networks: ', networks.keys())

    # Delete network from list
    if network_id in networks:
        del networks[network_id]

    print('Networks: ', networks.keys())

    # Save network to file
    save_networks_to_file(networks)

    return jsonify(networks)


if __name__ == '__main__':
    app.run(debug=True)
