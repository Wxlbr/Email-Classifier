import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_value):
        self.input = input_value
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative(self.input))
