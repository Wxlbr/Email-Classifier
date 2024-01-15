import os
import json
import random

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from main import Classifier

app = Flask(__name__)
socketio = SocketIO(app)

training_classifiers = {}
active_classifier = Classifier()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/edit-network', methods=['GET'])
def edit_network():

    # Get network Id from request
    network_id = request.args.get('networkId')

    # Get network from file
    with open(f'./inc/networks/{network_id}.json', 'r', encoding='utf-8') as f:
        network = json.load(f)

    print(network)

    return render_template('editNetwork.html', data=network)


@app.route('/save-new-network', methods=['POST'])
def save_network():

    # Get networks
    networks = [file for file in os.listdir('./inc/networks/') if file != "active"] + \
        os.listdir('./inc/networks/active/')

    # Generate new network id
    network_id = f'network-{random.randint(1000, 9999)}'
    while network_id in networks:
        network_id = f'network-{random.randint(1000, 9999)}'

    with open(f'./inc/networks/{network_id}.json', 'w', encoding='utf-8') as f:
        json.dump({
            "networkId": network_id,
            "name": f"Network {network_id.split('-')[0]}",
            "activeCard": False,
            "layers": {},
            "inputSize": 0,
            "outputSize": 1,
            "valid": False,
            "status": "inactive",
            "network": {}
        }, f, indent=4)

    return jsonify({'status': 'success'})


@app.route('/save-layers', methods=['POST'])
def save_layers():
    # Get data from request
    data = request.get_json()

    network_id = data.get('networkId')
    layers = data.get('layers')
    input_size = data.get('inputSize')
    output_size = data.get('outputSize')

    valid = all(layer['valid']
                for layer in layers.values()) if layers else False

    # Get network from file
    with open(f'./inc/networks/{network_id}.json', 'r', encoding='utf-8') as f:
        network = json.load(f)

    # Save network to list
    network.update({
        'layers': layers,
        'inputSize': input_size,
        'outputSize': output_size,
        'valid': valid
    })

    # Save network to file
    with open(f'./inc/networks/{network_id}.json', 'w', encoding='utf-8') as f:
        json.dump(network, f, indent=4)

    return jsonify(data)


@app.route('/switch-active-network', methods=['POST'])
def switch_active_card():

    network_id = request.get_json().get('newNetworkId')

    # Only one file in path ./inc/networks/active/ at a time, open it
    active_network_file = os.listdir('./inc/networks/active/')[0]

    os.rename(f'./inc/networks/active/{active_network_file}',
              f'./inc/networks/{active_network_file}')

    os.rename(f'./inc/networks/{network_id}.json',
              f'./inc/networks/active/{network_id}.json')

    return jsonify({'status': 'success'})


@app.route('/get-networks', methods=['GET'])
def get_networks():

    # Get networks from files
    networks = {'active': {}, 'inactive': {}}

    for file in os.listdir('./inc/networks/'):
        if file == 'active':
            # Ignore active folder directory
            continue

        with open(f'./inc/networks/{file}', 'r', encoding='utf-8') as f:
            networks['inactive'][file.split('.')[0]] = json.load(f)

    with open(f'./inc/networks/active/{os.listdir("./inc/networks/active/")[0]}', 'r', encoding='utf-8') as f:
        active_network = json.load(f)

    networks['active'][active_network['networkId']] = active_network

    return jsonify(networks)


@app.route('/get-network', methods=['POST'])
def get_network():

    # Get network Id from request
    network_id = request.get_json().get('networkId')

    # Get network from file
    with open(f'./inc/networks/{network_id}.json', 'r', encoding='utf-8') as f:
        network = json.load(f)

    return jsonify(network)


@socketio.on('connect', namespace='/train')
def handle_train_connect():
    print('Client connected to /train namespace')


@socketio.on('disconnect', namespace='/train')
def handle_train_disconnect():
    print('Client disconnected from /train namespace')


@app.route('/train-network', methods=['POST'])
def train_network():

    print('Received train request')

    network_id = request.get_json().get('networkId')
    epochs = request.get_json().get('epochs')

    # Get layers from file
    with open(f'./inc/networks/{network_id}.json', 'r', encoding='utf-8') as f:
        network = json.load(f)

    layers = network['layers']

    if 'network' in network:
        network['network'].clear()

    # Save network to file
    with open(f'./inc/networks/{network_id}.json', 'w', encoding='utf-8') as f:
        json.dump(network, f, indent=4)

    # thread = threading.Thread(target=train_network_thread, args=(
    #     network_id, layers, epochs,))
    # thread.start()

    training_classifiers[network_id] = Classifier()

    training_classifiers[network_id].start_training_thread(
        network_id, layers, epochs, socketio)

    return jsonify({'status': 'success'})


@app.route('/current-training', methods=['GET'])
def get_current_training_networks():

    ids = []

    for network_id in training_classifiers:
        if training_classifiers[network_id].get_is_training():
            ids.append(network_id)

    return jsonify({'status': 'success', 'ids': ids})


@app.route('/stop-training', methods=['POST'])
def stop_training():

    network_id = request.get_json().get('networkId')

    training_classifiers[network_id].stop_training()

    return jsonify({'status': 'success'})


@app.route('/delete-network', methods=['POST'])
def delete_network():

    network_id = request.get_json().get('networkId')

    # Check if network id is in path ./inc/networks/
    if os.path.exists(f'./inc/networks/{network_id}.json'):
        os.remove(f'./inc/networks/{network_id}.json')

    return jsonify({'status': 'success'})


@app.route('/toggle-active-network', methods=['POST'])
def toggle_active_network():

    activate = request.get_json().get('activate')

    # Get network from file
    with open(f'./inc/networks/active/{os.listdir("./inc/networks/active/")[0]}', 'r', encoding='utf-8') as f:
        network = json.load(f)

    if activate:

        print('activating')

        active_classifier.load_network(network['network'])

        # Change network status
        network['status'] = 'active'

        print('Loaded Network')

        # Start classification thread
        active_classifier.start_classification_thread()

    else:

        print('deactivating')

        active_classifier.stop_classification()

        # Change network status
        network['status'] = 'inactive'

    # Save network to file
    with open(f'./inc/networks/active/{os.listdir("./inc/networks/active/")[0]}', 'r', encoding='utf-8') as f:
        json.dump(network, f, indent=4)

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    socketio.run(app, debug=True)
