from calc import mean, log


class MSE():
    def calc(self, actual: int, pred: list) -> float:
        return mean([[(actual - p[0]) ** 2] for p in pred])

    def derivative(self, actual, pred):
        return [[2 * (p[0] - actual)] for p in pred]


class BinaryCrossEntropy():
    def calc(self, actual, pred):
        return mean([[-actual * log(p[0]) - (1 - actual) * log(1 - p[0])] for p in pred])

    def derivative(self, actual, pred):
        return ([[(1 - actual) / (1 - p[0]) - actual / p[0]] for p in pred])
