from activation import Sigmoid
from calc import dot, multiply, transpose, randn, zeros, add_matrices, sub_matrices


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
        self.weights -= multiply(weights_gradient, learning_rate)
        self.bias -= multiply(output_gradient, learning_rate)
        return input_gradient


class Recurrent():
    '''
    Recurrent layer (fully connected layer with recurrent connections)

    input_size: size of input vector
    output_size: size of output vector
    activation: activation function
    '''

    def __init__(self, input_size, output_size, activation=Sigmoid()):
        # Initialize input and output
        self.input = None
        self.output = None

        # Initialize sizes of input and output
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights and biases to random values
        self.weights = randn(output_size, input_size)
        self.recurrent_weights = randn(output_size, output_size)
        self.bias = randn(output_size, 1)

        # Initialize hidden state to zero vectors
        self.hidden_state = zeros(output_size, 1)

        # Initialize activation function
        self.activation = activation

    def forward(self, input_value):
        '''
        Forward pass of the layer
        '''

        # Save input value
        self.input = input_value

        # Update hidden state
        self.hidden_state = dot(self.recurrent_weights, self.hidden_state)

        # Calculate the weighted input and hidden state
        weighted_input = dot(self.weights, self.input)
        weighted_hidden = self.hidden_state

        # Update output by adding weighted input and hidden state
        self.output = add_matrices(weighted_input, weighted_hidden)

        # Apply activation function
        self.output = self.activation.forward(self.output)

        # Update hidden state for the next iteration
        self.hidden_state = self.output

        # Return output
        return self.output

    def backward(self, output_gradient, learning_rate):
        '''
        Backward pass of the layer
        '''

        # Apply activation function derivative
        output_gradient = self.activation.backward(
            output_gradient)

        # Calculate gradients
        weights_gradient = dot(output_gradient, transpose(self.input))
        recurrent_weights_gradient = dot(
            output_gradient, transpose(self.hidden_state))
        input_gradient = dot(transpose(self.weights), output_gradient)

        # Update weights and biases
        self.weights = sub_matrices(self.weights, multiply(
            weights_gradient, learning_rate))

        self.recurrent_weights = sub_matrices(self.recurrent_weights, multiply(
            recurrent_weights_gradient, learning_rate))

        self.bias = sub_matrices(self.bias, multiply(
            output_gradient, learning_rate))

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
