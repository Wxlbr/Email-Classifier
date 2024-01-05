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
        # Initialise input
        self.input = None

        # Initialise weights and biases to random values
        self.weights = randn(output_size, input_size)
        self.bias = randn(output_size, 1)

        # Initialise activation function
        self.activation = activation

    def forward(self, input_value):

        # Save input value
        self.input = input_value

        # Calculate the weighted input
        weighted_input = dot(self.weights, self.input)

        # Calculate output by adding weighted input and bias
        output = add_matrices(weighted_input, self.bias)

        # Apply activation function
        output = self.activation.forward(output)

        # Return output
        return output

    def backward(self, output_gradient, learning_rate):

        # Apply activation function derivative
        output_gradient = self.activation.backward(
            output_gradient, learning_rate)

        # Calculate gradients
        weights_gradient = dot(output_gradient, transpose(self.input))
        input_gradient = dot(transpose(self.weights), output_gradient)

        # Update weights and biases
        self.weights = sub_matrices(self.weights, multiply(
            weights_gradient, learning_rate))
        self.bias = sub_matrices(self.bias, multiply(
            output_gradient, learning_rate))

        # Return gradient with respect to input
        return input_gradient

    def info(self):
        '''
        Return a dictionary containing the variables of the layer
        '''

        return {
            'input_size': len(self.input),
            'output_size': len(self.weights),
            'weights': self.weights,
            'bias': self.bias,
            'activation': self.activation.get_type(),
            'type': 'dense'
        }

    def load(self, info):
        '''
        Load the variables of the layer from a dictionary
        '''

        self.weights = info.get('weights', self.weights)
        self.bias = info.get('bias', self.bias)


class Recurrent():
    '''
    Recurrent layer (fully connected layer with recurrent connections)

    input_size: size of input vector
    output_size: size of output vector
    activation: activation function
    '''

    def __init__(self, input_size, output_size, activation=Sigmoid()):
        # Initialise input
        self.input = None

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

        # Save input value for the backward pass
        self.input = input_value

        # Update hidden state
        self.hidden_state = dot(self.recurrent_weights, self.hidden_state)

        # Calculate the weighted input
        weighted_input = dot(self.weights, self.input)

        # Update output by adding weighted input and bias
        output = add_matrices(weighted_input, self.bias)

        # Update output by adding weighted input and hidden state
        output = add_matrices(weighted_input, self.hidden_state)

        # Apply activation function
        output = self.activation.forward(output)

        # Update hidden state for the next iteration
        self.hidden_state = output

        # Return output
        return output

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
