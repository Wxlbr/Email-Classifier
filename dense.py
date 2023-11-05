import numpy as np
from scipy import signal

from activation import Sigmoid
from layer import Layer
from matrix import dot, mul

def convolve2d(input_array, kernel_array, mode="full"):
    # Testing as signal is faster
    return signal.convolve2d(input_array, kernel_array, mode=mode)

    if mode not in ["full", "valid"]:
        raise ValueError("Mode must be one of 'full' or 'valid'")

    input_height, input_width = input_array.shape
    kernel_height, kernel_width = kernel_array.shape
    if mode == "full":
        pad_height = kernel_height - 1
        pad_width = kernel_width - 1
        output_height = input_height + pad_height
        output_width = input_width + pad_width
        input_array = np.pad(input_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    elif mode == "valid":
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(input_array[y:y + kernel_height, x:x + kernel_width] * kernel_array)

    return output

class Dense(Layer):
    '''
    Dense layer (fully connected layer)

    input_size: size of input vector
    output_size: size of output vector
    activation: activation function
    '''

    def __init__(self, input_size, output_size, activation=Sigmoid()):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.activation = activation

    def forward(self, input_value):
        self.input = input_value
        self.output = dot(self.weights, self.input) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient, learning_rate)
        weights_gradient = dot(output_gradient, self.input.T)
        input_gradient = dot(self.weights.T, output_gradient)
        self.weights -= mul(weights_gradient, learning_rate)
        self.bias -= mul(output_gradient, learning_rate)
        return input_gradient

class Convolutional(Layer):
    '''
    Convolutional layer (2D convolution)

    input_shape: (depth, height, width)
        e.g. [[1, 2, 3], [4, 5, 6]] -> depth = 1, height = 2, width = 3
    kernel_size: size of kernel (square)
    depth: number of kernels
    activation: activation function
    '''

    def __init__(self, input_shape, kernel_size, depth, activation=Sigmoid()):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        self.activation = activation

    def forward(self, input_value):
        self.input = input_value
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += convolve2d(self.input[j], self.kernels[i, j], "valid")
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient, learning_rate)

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = convolve2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Recurrent(Layer):
    '''
    Recurrent layer (fully connected layer with recurrent connections)

    input_size: size of input vector
    output_size: size of output vector
    activation: activation function
    '''

    def __init__(self, input_size, output_size, activation=Sigmoid()):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.recurrent_weights = np.random.randn(output_size, output_size)
        self.bias = np.random.randn(output_size, 1)
        self.activation = activation

    def forward(self, input_value):
        self.input = input_value
        self.output = np.copy(self.bias)
        self.output = dot(self.weights, self.input) + dot(self.recurrent_weights, self.output) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient, learning_rate)
        weights_gradient = dot(output_gradient, self.input.T)
        recurrent_weights_gradient = dot(output_gradient, self.output.T)
        input_gradient = dot(self.weights.T, output_gradient)
        self.weights -= mul(weights_gradient, learning_rate)
        self.recurrent_weights -= mul(recurrent_weights_gradient, learning_rate)
        self.bias -= mul(np.sum(output_gradient, axis=1, keepdims=True), learning_rate)
        return input_gradient