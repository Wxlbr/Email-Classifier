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