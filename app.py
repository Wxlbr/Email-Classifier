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

# TODO: Swap queue for a websocket


app = Flask(__name__)
queue = Queue()
history_queue = Queue()
threads = {}


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
    if network_id not in networks:
        networks[network_id] = {}
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


@app.route('/toggle-active-network', methods=['POST'])
def toggle_active_network():
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
        threads[network_id] = (
            {'networkId': network_id, 'thread': thread, 'stop_event': stop_event})

    else:

        # TODO: Better handling of stopping threads

        print('deactivating')

        # Get active thread
        for thread in threads.values():
            if thread['networkId'] == network_id:
                thread['stop_event'].set()
                thread['thread'].join()

        # Change network status
        networks[network_id]['status'] = 'inactive'

    # Save network to file
    save_networks_to_file(networks)

    return jsonify(networks)


if __name__ == '__main__':
    app.run(debug=True)
