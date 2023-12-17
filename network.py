import time
import json

from loss import MSE, BinaryCrossEntropy
from layer import Recurrent
from activation import Sigmoid, Tanh


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

        return correct / len(x_test) * 100

    def train(self, x_train, y_train, epochs=100, learning_rate=0.01, loss='binary_crossentropy', validation_data=None, verbose=True, queue=None):

        assert self.layers, 'Network has no layers.'

        assert len(x_train) == len(
            y_train), "Training data and labels must be of same length."

        # select loss function
        assert loss in self.LOSS_TYPES, "Unknown loss function."
        loss = self.LOSS_TYPES[loss]()

        if validation_data:
            x_val, y_val = validation_data

        if queue:
            queue.put({"data": {
                "epoch": 0,
                "epochs": epochs,
                "error": 0,
                "accuracy": 0,
                "eta": 0,
            }})

        # training loop
        for e in range(epochs):
            error = 0
            start = time.time()
            count = 0
            for x, y in zip(x_train, y_train):
                count += 1
                # forward
                output = self.predict(x)

                # error
                # print(y, output)
                error += loss.calc(y, output)

                # backward
                gradient = loss.derivative(y, output)
                # print(gradient)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
                    # print(gradient)

                # if queue and count % (int(len(x_train) * 0.1)) == 0:
                #     queue.put({"data": {
                #     "epoch": e + 1,
                #     "epochs": epochs,
                #     "error": error / len(x_train),
                #     "accuracy": self.accuracy(x_val, y_val),
                #     "eta": ((time.time() - start) / (e + 1)) * (epochs - e - 1),
                # }})

            if queue:
                queue.put({"data": {
                    "epoch": e + 1,
                    "epochs": epochs,
                    "error": float(f"{error / len(x_train):.4f}"),
                    "accuracy": float(f"{self.accuracy(x_val, y_val):.2f}"),
                    "eta": float(f"{((time.time() - start) / (e + 1)) * (epochs - e - 1):.2f}"),
                }})

            if verbose:
                print(f"{e + 1}/{epochs}, error={error / len(x_train):.4f}", end="")

                if validation_data:
                    print(
                        f", val_accuracy={self.accuracy(x_val, y_val):.4f}%", end="")

                print(f", duration={time.time() - start:.2f}s", end="")

                print()

        if queue:
            queue.put({"data": "done"})

    def info(self):
        return {i: layer.info() for i, layer in enumerate(self.layers)}

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.info(), f, indent=4)

    def load(self, path):
        self.layers.clear()

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

        # TODO: Key checks
        input_size = int(layer['inputSize'])
        output_size = int(layer['outputSize'])

        if not self.layers:
            self.layers = []

        self.layers.append(self.LAYER_TYPES[layer['type']](
            input_size, output_size, activation=self.ACTIVATION_TYPES[layer['activation']]()))
