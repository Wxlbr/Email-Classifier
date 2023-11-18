import numpy as np

class Activation():
    def __init__(self, activation, activation_derivative):
        self.input = None
        self.output = None
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_value):
        self.input = input_value
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # print(output_gradient.shape, self.activation_derivative(self.input).shape)
        return np.multiply(output_gradient, self.activation_derivative(self.input))

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_derivative)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative)

class Softmax():
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_value):
        tmp = np.exp(input_value)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
