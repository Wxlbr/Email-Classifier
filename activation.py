import numpy as np


class Tanh():
    def __init__(self):
        self.input = None
        self.output = None
        self.type = 'tanh'

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        self.input = x
        return self.tanh(x)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.tanh_derivative(self.input))

    def get_type(self):
        return self.type


class Sigmoid():
    def __init__(self):
        self.input = None
        self.output = None
        self.type = 'sigmoid'

    def sigmoid(self, x):
        return [[1 / (1 + np.exp(-value)) for value in row] for row in x]

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, input_value):
        self.input = input_value
        return self.sigmoid(self.input)

    def backward(self, output_gradient, learning_rate):
        # print(output_gradient.shape, self.activation_derivative(self.input).shape)
        return np.multiply(output_gradient, self.sigmoid_derivative(self.input))

    def get_type(self):
        return self.type
