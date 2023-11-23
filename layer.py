import numpy as np

from activation import Sigmoid
from matrix import dot, mul

class Dense():
    '''
    Dense layer (fully connected layer)

    input_size: size of input vector
    output_size: size of output vector
    activation: activation function
    '''

    def __init__(self, input_size, output_size, activation=Sigmoid()):
        self.input = None
        self.output = None
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

class Recurrent():
    '''
    Recurrent layer (fully connected layer with recurrent connections)

    input_size: size of input vector
    output_size: size of output vector
    activation: activation function
    '''

    def __init__(self, input_size, output_size, activation=Sigmoid()):
        # Initialise input and output
        self.input = None
        self.output = None

        # Initialise sizes of input and output
        self.input_size = input_size
        self.output_size = output_size

        # Initialise weights and biases to random values
        self.weights = np.random.randn(output_size, input_size).tolist()
        self.recurrent_weights = np.random.randn(output_size, output_size).tolist()
        self.bias = np.random.randn(output_size, 1).tolist()

        # Initialise hidden state to zero vectors
        self.hidden_state = np.zeros((output_size, 1)).tolist()

        # Initialise activation function
        self.activation = activation

    def forward(self, input_value):
        '''
        Forward pass of the layer
        '''

        # Save input value
        self.input = input_value

        # Initialise output as a copy of the bias
        self.output = np.copy(self.bias).tolist()

        # Update hidden state
        self.hidden_state = dot(self.recurrent_weights, self.hidden_state)

        # Add weighted input and hidden state to output
        # self.output = dot(self.weights, self.input) + self.hidden_state + self.bias

        # TODO: Temporary fix for matrix multiplication
        self.output = [[dot(self.weights, self.input)[i] + self.hidden_state[i][0] + self.bias[i][0]] for i in range(self.output_size)]

        # Apply activation function
        self.output = self.activation.forward(self.output)

        # Return output
        return self.output

    def backward(self, output_gradient, learning_rate):
        '''
        Backward pass of the layer
        '''

        # Apply activation function derivative
        output_gradient = self.activation.backward(output_gradient, learning_rate)

        # Calculate gradients
        weights_gradient = dot(output_gradient, self.input.T)
        recurrent_weights_gradient = dot(output_gradient, self.hidden_state.T)
        input_gradient = dot(self.weights.T, output_gradient)

        # Update weights and biases
        self.weights -= mul(weights_gradient, learning_rate)
        self.recurrent_weights -= mul(recurrent_weights_gradient, learning_rate)
        self.bias -= mul(np.sum(output_gradient, axis=1, keepdims=True), learning_rate)

        # Return gradient with respect to input
        return input_gradient

    # TODO: Remove numpy dependency
    def info(self):
        '''
        Return a dictionary containing the variables of the layer
        '''

        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'weights': self.weights,
            'recurrent_weights': self.recurrent_weights,
            'bias': self.bias,
        }

    def load(self, info):
        '''
        Load the variables of the layer from a dictionary
        '''

        self.weights = info.get('weights', self.weights)
        self.recurrent_weights = info.get('recurrent_weights', self.recurrent_weights)
        self.bias = info.get('bias', self.bias)