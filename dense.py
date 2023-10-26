import numpy as np
from scipy import signal
from activation import Tanh, Sigmoid, Softmax
from layer import Layer

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
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= weights_gradient * learning_rate
        self.bias -= output_gradient * learning_rate
        return input_gradient

class Convolutional(Layer):
    '''
    Convolutional layer (2D convolution)

    input_shape: (depth, height, width)
        e.g. [[1, 2, 3], [4, 5, 6]] -> depth = 1, height = 2, width = 3
    kernel_size: size of kernel (square)
    depth: number of kernels
    '''

    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_value):
        self.input = input_value
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                # print(self.input[j], self.kernels[i, j], "valid")
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient