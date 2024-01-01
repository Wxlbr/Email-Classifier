import os
import json
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from main import Classifier

app = Flask(__name__)
socketio = SocketIO(app)


def save_networks_to_file(networks):
    with open('./inc/networks.json', 'w', encoding='utf-8') as f:
        json.dump(networks, f, indent=4)


def load_networks_from_file():
    if not os.path.exists('./inc/networks.json'):
        return {}
    with open('./inc/networks.json', 'r', encoding='utf-8') as f:
        return json.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/edit-network', methods=['GET'])
def edit_network():

    # Get network Id from request
    network_id = request.args.get('networkId')

    # Get network from list
    networks = load_networks_from_file()

    # Add network id to network
    networks[network_id]['networkId'] = network_id

    return render_template('index2.html', data=networks[network_id])


@app.route('/save-network', methods=['POST'])
def save_network():
    # Get data from request
    # print(request.data)

    data = request.get_json()

    network_id = data.get('networkId')

    # Get networks from file
    networks = load_networks_from_file()

    networks[network_id] = {
        "networkId": network_id,
        "name": f"Network {network_id.split('-')[0]}",
        "activeCard": False,
        "layers": {},
        "inputSize": 0,
        "outputSize": 0,
        "valid": False,
        "status": "inactive",
        "network": {}
    }

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
    networks[network_id] = {
        'layers': layers,
        'inputSize': input_size,
        'outputSize': output_size,
        'activeCard': False,
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


@socketio.on('connect', namespace='/train')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect', namespace='/train')
def handle_disconnect():
    print('Client disconnected')


@app.route('/train-network', methods=['POST'])
def train_network():

    data = request.get_json()

    network_id = data.get('networkId')
    network = data.get('network')

    thread = threading.Thread(target=train_network_thread, args=(
        network_id, network,))
    thread.start()

    return jsonify({'status': 'success'})


def train_network_thread(network_id, network):
    classifier = Classifier()

    for layer in network['layers'].values():
        classifier.add_layer(layer['layerConfig'])

    classifier.train_network(socketio=socketio, netId=network_id)

    # Get networks from file
    networks = load_networks_from_file()

    if network_id not in networks:
        networks[network_id] = {}
    networks[network_id]['network'] = classifier.net.info()

    save_networks_to_file(networks)


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


@app.route('/toggle-active-network', methods=['POST'])
def toggle_active_network():

    # TODO: Pass websocket to thread so it can emit events

    data = request.get_json()

    activate = data.get('activate')

    # Get networks from file
    networks = load_networks_from_file()

    # Get active network id
    network_id = next(
        (network_id for network_id in networks if networks[network_id]['activeCard']), None)

    if activate:

        print('activating')

        classifier = Classifier()

        with open('./inc/temp_model_loading.json', 'w', encoding='utf-8') as f:
            json.dump(networks[network_id]['network'], f, indent=4)

        classifier.load_network('./inc/temp_model_loading.json')

        # Delete temp file
        os.remove('./inc/temp_model_loading.json')

        # Change network status
        networks[network_id]['status'] = 'active'

        # Create stop event
        stop_event = threading.Event()

        # Start thread with classifier
        thread = threading.Thread(
            target=classifier.main, args=(True, -1, stop_event))

        thread.start()

        # Save thread to queue
        # threads[network_id] = (
        #     {'networkId': network_id, 'thread': thread, 'stop_event': stop_event})

    else:

        # TODO: Better handling of stopping threads

        print('deactivating')

        # Get active thread
        # for thread in threads.values():
        #     if thread['networkId'] == network_id:
        #         thread['stop_event'].set()
        #         thread['thread'].join()

        # Change network status
        networks[network_id]['status'] = 'inactive'

    # Save network to file
    save_networks_to_file(networks)

    return jsonify(networks)


if __name__ == '__main__':
    socketio.run(app, debug=True)
