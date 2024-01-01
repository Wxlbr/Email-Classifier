import time
import json

from loss import MSE, BinaryCrossEntropy
from layer import Recurrent
from activation import Sigmoid, Tanh
from calc import mean, convert_time


class Network:

    LAYER_TYPES = {'recurrent': Recurrent}
    ACTIVATION_TYPES = {'sigmoid': Sigmoid, 'tanh': Tanh}
    LOSS_TYPES = {'mse': MSE, 'binary_crossentropy': BinaryCrossEntropy}

    def __init__(self, layers=[]):
        self.layers = layers

    def predict(self, input_value):
        assert self.layers, 'Network has no layers.'

        output = input_value
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def accuracy(self, x_test, y_test):
        correct = 0
        for x, y in zip(x_test, y_test):
            pred = self.predict(x)[0][0] > 0.5
            if y == pred:
                correct += 1

        return (correct * 100) / len(x_test)

    def train(self, x_train, y_train, epochs=2, learning_rate=0.01, loss='binary_crossentropy', validation_data=None, verbose=True, socketio=None, netId=None):

        assert self.layers, 'Network has no layers.'

        assert len(x_train) == len(
            y_train), "Training data and labels must be of same length."

        # select loss function
        assert loss in self.LOSS_TYPES, "Unknown loss function."
        loss = self.LOSS_TYPES[loss]()

        if socketio:

            socketio.emit('training_update', {
                "status": "training",
                "epoch": "0",
                "epochs": epochs,
                "error": "0",
                "accuracy": "-",
                "epochEta": "0s",
                "totalEta": "0s",
                "networkId": netId
            }, namespace='/train')

        # Training loop
        for e in range(1, epochs+1):
            error = 0
            inner_durations = []

            # Loop through all training data
            for i, (x, y) in enumerate(zip(x_train, y_train)):

                # Start timer
                inner_start = time.time()

                # Forward propagation
                output = self.predict(x)

                # Calculate error
                error += loss.calc(y, output)

                # Backward propagation
                gradient = loss.derivative(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

                # Stop timer and save duration
                inner_durations.append(time.time() - inner_start)

                # Update progress bar
                if socketio and i % 10 == 0:

                    epochEta = mean(inner_durations) * (len(x_train) - (i+1))
                    totalEta = mean(inner_durations) * \
                        (epochs - e) * len(x_train) + epochEta

                    socketio.emit('training_update', {
                        "status": "training",
                        "epoch": e,
                        "epochs": epochs,
                        "error": f"{error / (i+1):.4f}",
                        "accuracy": "-",
                        "epochEta": convert_time(epochEta),
                        "totalEta": convert_time(totalEta),
                        "networkId": netId
                    }, namespace='/train')

            if socketio:

                socketio.emit('training_update', {
                    "status": "training",
                    "epoch": e,
                    "epochs": epochs,
                    "error": f"{error / len(x_train):.4f}",
                    "accuracy": f"{self.accuracy(validation_data[0], validation_data[1]):.4f}",
                    "epochEta": "0s",
                    "totalEta": convert_time(mean(inner_durations) * (epochs - e)),
                    "networkId": netId
                }, namespace='/train')

            if verbose:
                print(f"{e}/{epochs}, error={error / len(x_train):.4f}", end="")

                if validation_data:
                    print(
                        f", val_accuracy={self.accuracy(validation_data[0], validation_data[1]):.4f}%", end="")

                print(f", duration={sum(inner_durations):.2f}s", end="")

                print()

        if socketio:

            socketio.emit('training_done', {
                "status": "done",
                "networkId": netId
            }, namespace='/train')

    def info(self):
        return {i: layer.info() for i, layer in enumerate(self.layers)}

    def save(self, path):
        # TODO: Check path is valid
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.info(), f, indent=4)

    def load(self, path):
        self.layers.clear()

        # TODO: Check path exists and is valid
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i in data:
            assert data[i]['type'] in self.LAYER_TYPES, "Unknown layer type"
            assert data[i]['activation'] in self.ACTIVATION_TYPES, "Unknown activation type"

            layer = self.LAYER_TYPES[data[i]['type']](
                data[i]['input_size'], data[i]['output_size'], activation=self.ACTIVATION_TYPES[data[i]['activation']]())
            layer.load(data[i])
            self.layers.append(layer)

    def add_layer(self, layer):
        '''
        Add a layer to the network
        '''

        assert layer['type'] in self.LAYER_TYPES, "Unknown layer type"
        assert layer['activation'] in self.ACTIVATION_TYPES, "Unknown activation type"

        assert layer['inputSize'].isdigit(), "inputSize must be an integer"
        assert layer['outputSize'].isdigit(), "outputSize must be an integer"

        input_size = int(layer['inputSize'])
        output_size = int(layer['outputSize'])

        if not self.layers:
            self.layers = []

        self.layers.append(self.LAYER_TYPES[layer['type']](
            input_size, output_size, activation=self.ACTIVATION_TYPES[layer['activation']]()))
