import random

import numpy as np

from activation import Sigmoid
from matrix import dot, mul, transpose, randn, zeros


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
        self.weights = randn(output_size, input_size)
        self.bias = randn(output_size, 1)
        self.activation = activation

    def forward(self, input_value):
        self.input = input_value
        self.output = dot(self.weights, self.input) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(
            output_gradient, learning_rate)
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
        self.weights = randn(output_size, input_size)
        self.recurrent_weights = randn(output_size, output_size)
        self.bias = randn(output_size, 1)

        # Initialise hidden state to zero vectors
        self.hidden_state = zeros(output_size, 1)

        # Initialise activation function
        self.activation = activation

    def forward(self, input_value):
        '''
        Forward pass of the layer
        '''

        # Save input value
        self.input = input_value

        # Initialise output as a copy of the bias
        self.output = [[x] for x in sum(self.bias, [])]

        # Update hidden state
        self.hidden_state = dot(self.recurrent_weights, self.hidden_state)

        # Add weighted input and hidden state to output
        # TODO: Temporary fix for matrix multiplication
        # self.output = dot(self.weights, self.input) + self.hidden_state + self.bias

        for i in range(self.output_size):
            for j in range(self.input_size):
                self.output[i][0] += self.weights[i][j] * self.input[j]
            self.output[i][0] += self.hidden_state[i][0] + self.bias[i][0]

        # Apply activation function
        self.output = self.activation.forward(self.output)

        # Return output
        return self.output

    def backward(self, output_gradient, learning_rate):
        '''
        Backward pass of the layer
        '''

        # Apply activation function derivative
        output_gradient = self.activation.backward(
            output_gradient, learning_rate)

        # Calculate gradients
        weights_gradient = dot(output_gradient, transpose(self.input))
        recurrent_weights_gradient = dot(
            output_gradient, transpose(self.hidden_state))
        input_gradient = dot(transpose(self.weights), output_gradient)

        # Update weights and biases
        self.weights -= mul(weights_gradient, learning_rate)
        self.recurrent_weights -= mul(recurrent_weights_gradient,
                                      learning_rate)
        self.bias -= mul([[sum(row, keepdims=True)]
                         for row in output_gradient], learning_rate)

        # Return gradient with respect to input
        return input_gradient

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
            'activation': self.activation.get_type(),
            'type': 'recurrent'
        }

    def load(self, info):
        '''
        Load the variables of the layer from a dictionary
        '''

        self.weights = info.get('weights', self.weights)
        self.recurrent_weights = info.get(
            'recurrent_weights', self.recurrent_weights)
        self.bias = info.get('bias', self.bias)
