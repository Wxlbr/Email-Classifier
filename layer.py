import numpy as np
from scipy import signal

from activation import Sigmoid
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

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_value):
        # Return output
        pass

    def backward(self, output_gradient, learning_rate):
        # Update parameters and return input gradient
        pass

    def save(self, file):
        # Save layer parameters
        pass

    def load(self, file):
        # Load layer parameters
        pass

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
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernel = np.random.randn(*self.kernel_shape)
        self.bias = np.random.randn(*self.output_shape)
        self.activation = activation

    def forward(self, input_value):
        self.input = input_value
        self.output = np.copy(self.bias)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += convolve2d(self.input[j], self.kernel[i, j], "valid")
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient, learning_rate)

        kernel_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradient[i, j] = convolve2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += convolve2d(output_gradient[i], self.kernel[i, j], "full")

        self.kernel -= learning_rate * kernel_gradient
        self.bias -= learning_rate * output_gradient
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
        self.hidden_state = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, input_value):
        self.input = input_value
        self.output = np.copy(self.bias)
        self.hidden_state = dot(self.recurrent_weights, self.hidden_state)
        self.output = dot(self.weights, self.input) + self.hidden_state + self.bias
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient, learning_rate)
        weights_gradient = dot(output_gradient, self.input.T)
        recurrent_weights_gradient = dot(output_gradient, self.hidden_state.T)
        input_gradient = dot(self.weights.T, output_gradient)
        self.weights -= mul(weights_gradient, learning_rate)
        self.recurrent_weights -= mul(recurrent_weights_gradient, learning_rate)
        self.bias -= mul(np.sum(output_gradient, axis=1, keepdims=True), learning_rate)
        return input_gradient

# class tempSigmoid:
#     def forward(self, x):
#         return 1 / (1 + np.exp(-x))

#     def backward(self, output_gradient, learning_rate):
#         return output_gradient * (1 - output_gradient)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTM(Layer):
    def __init__(self, input_size, output_size, hidden_size=100, activation=Sigmoid()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        self.w_f = np.random.randn(hidden_size, hidden_size + input_size)
        self.w_i = np.random.randn(hidden_size, hidden_size + input_size)
        self.w_c = np.random.randn(hidden_size, hidden_size + input_size)
        self.w_o = np.random.randn(hidden_size, hidden_size + input_size)
        self.w_y = np.random.randn(output_size, hidden_size)

        self.b_f = np.random.randn(hidden_size, 1)
        self.b_i = np.random.randn(hidden_size, 1)
        self.b_c = np.random.randn(hidden_size, 1)
        self.b_o = np.random.randn(hidden_size, 1)
        self.b_y = np.random.randn(output_size, 1)

        self.f = np.zeros((hidden_size, 1))
        self.i = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))
        self.o = np.zeros((hidden_size, 1))
        self.h = np.zeros((hidden_size, 1))

    def forward(self, input_value, hidden_state, cell_state):
        self.input = input_value
        self.h = hidden_state

        self.f = sigmoid(self.w_f @ np.vstack((self.h, self.input)) + self.b_f)
        self.i = sigmoid(self.w_i @ np.vstack((self.h, self.input)) + self.b_i)
        self.c = np.tanh(self.w_c @ np.vstack((self.h, self.input)) + self.b_c)
        self.o = sigmoid(self.w_o @ np.vstack((self.h, self.input)) + self.b_o)

        self.cell_state = self.f * cell_state + self.i * self.c
        self.h = self.o * np.tanh(self.cell_state)

        self.output = self.w_y @ self.h + self.b_y
        self.output = sigmoid(self.output)

        return self.output, self.h, self.cell_state

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient, learning_rate)

        self.output_gradient = output_gradient

        self.w_y_gradient = self.output_gradient @ self.h.T
        self.b_y_gradient = self.output_gradient

        self.h_gradient = self.w_y.T @ self.output_gradient
        self.o_gradient = self.h_gradient * np.tanh(self.cell_state)
        self.cell_state_gradient = self.h_gradient * self.o * (1 - np.tanh(self.cell_state) ** 2)
        self.c_gradient = self.cell_state_gradient * self.i
        self.i_gradient = self.cell_state_gradient * self.c
        self.f_gradient = self.cell_state_gradient * self.cell_state

        self.w_o_gradient = self.o_gradient @ np.vstack((self.h, self.input)).T
        self.b_o_gradient = self.o_gradient

        self.w_c_gradient = self.c_gradient @ np.vstack((self.h, self.input)).T
        self.b_c_gradient = self.c_gradient

        self.w_i_gradient = self.i_gradient @ np.vstack((self.h, self.input)).T
        self.b_i_gradient = self.i_gradient

        self.w_f_gradient = self.f_gradient @ np.vstack((self.h, self.input)).T
        self.b_f_gradient = self.f_gradient

        self.h_gradient = self.w_o.T @ self.o_gradient + self.w_c.T @ self.c_gradient + self.w_i.T @ self.i_gradient + self.w_f.T @ self.f_gradient
        self.input_gradient = self.w_o.T @ self.o_gradient + self.w_c.T @ self.c_gradient + self.w_i.T @ self.i_gradient + self.w_f.T @ self.f_gradient

        self.w_y -= mul(self.w_y_gradient, learning_rate)
        self.b_y -= mul(self.b_y_gradient, learning_rate)

        self.w_o -= mul(self.w_o_gradient, learning_rate)
        self.b_o -= mul(self.b_o_gradient, learning_rate)

        self.w_c -= mul(self.w_c_gradient, learning_rate)
        self.b_c -= mul(self.b_c_gradient, learning_rate)

        self.w_i -= mul(self.w_i_gradient, learning_rate)
        self.b_i -= mul(self.b_i_gradient, learning_rate)

        self.w_f -= mul(self.w_f_gradient, learning_rate)
        self.b_f -= mul(self.b_f_gradient, learning_rate)

        return self.input_gradient, self.h_gradient, self.cell_state_gradient