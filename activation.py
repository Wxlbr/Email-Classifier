from calc import multiply, exp, dot


class Tanh:
    def __init__(self):
        self.input = None
        self.output = None
        self.type = 'tanh'

    def tanh(self, x):
        return (exp(2*x) - 1) / (exp(2*x) + 1)

    def tanh_derivative(self, x):
        return 1 - self.tanh(x) ** 2

    def forward(self, x):
        self.input = x
        return self.tanh(x)

    def backward(self, x):
        return multiply(x, self.tanh_derivative(self.input))

    def get_type(self):
        return self.type


class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None
        self.type = 'sigmoid'

    def sigmoid(self, x):
        return [[1 / (1 + exp(-value)) for value in row] for row in x]

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return [[(1 - value) * value for value in row] for row in s]

    def forward(self, x):
        self.input = x
        return self.sigmoid(x)

    def backward(self, x):
        return dot(x, self.sigmoid_derivative(self.input))

    def get_type(self):
        return self.type
