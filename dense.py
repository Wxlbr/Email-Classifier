import numpy as np
from layer import Layer
from matrix import dot_product

class Dense(Layer):
    '''
    Dense layer (fully connected layer)
    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input_value):
        self.input = input_value
        return dot_product(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = dot_product(output_gradient, self.input.T)
        input_gradient = dot_product(self.weights.T, output_gradient)
        print("weights_gradient", weights_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
