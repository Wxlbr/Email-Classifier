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

    def train(self, x_train, y_train, epochs=1000, learning_rate=0.01, loss='binary_crossentropy', validation_data=None, verbose=True):

        assert self.layers, 'Network has no layers.'

        assert len(x_train) == len(
            y_train), "Training data and labels must be of same length."

        # select loss function
        assert loss in self.LOSS_TYPES, "Unknown loss function."
        loss = self.LOSS_TYPES[loss]()

        # training loop
        for e in range(epochs):
            error = 0
            start = time.time()
            for x, y in zip(x_train, y_train):
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

            if verbose:
                print(f"{e + 1}/{epochs}, error={error / len(x_train):.4f}", end="")

                if validation_data:
                    x_val, y_val = validation_data
                    print(
                        f", val_accuracy={self.accuracy(x_val, y_val):.4f}%", end="")

                print(f", duration={time.time() - start:.2f}s", end="")

                print()

    def save(self, path):
        data = {i: layer.info() for i, layer in enumerate(self.layers)}

        # print(data)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

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
