class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_value):
        # Return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: Optimiser
        # Update parameters and return input gradient
        pass