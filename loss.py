import numpy as np

class Loss:
    def __init__(self, loss, loss_derivative):
        self.loss_func = loss
        self.loss_derivative_func = loss_derivative

    def calc(self, y_true, y_pred):
        return self.loss_func(y_true, y_pred)

    def derivative(self, y_true, y_pred):
        return self.loss_derivative_func(y_true, y_pred)

class MSE(Loss):
    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))

        def mse_derivative(y_true, y_pred):
            return 2 * (y_pred - y_true) / np.size(y_true)

        super().__init__(mse, mse_derivative)

class BinaryCrossEntropy(Loss):
    def __init__(self):
        def binary_cross_entropy(y_true, y_pred):
            return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

        def binary_cross_entropy_derivative(y_true, y_pred):
            return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

        super().__init__(binary_cross_entropy, binary_cross_entropy_derivative)