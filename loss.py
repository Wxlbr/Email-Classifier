import numpy as np


class MSE():
    def calc(self, actual, pred):
        return np.mean(np.power(actual - pred, 2))

    def derivative(self, actual, pred):
        return 2 * (pred - actual) / np.size(actual)


class BinaryCrossEntropy():
    def calc(self, actual, pred):
        return np.mean(-actual * np.log(pred) - (1 - actual) * np.log(1 - pred))

    def derivative(self, actual, pred):
        return ((1 - actual) / (1 - pred) - actual / pred) / np.size(actual)
