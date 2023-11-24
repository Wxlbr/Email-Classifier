import numpy as np

def dot(x: list, y: list) -> list:
    # For testing purposes as numpy is faster
    return np.dot(np.array(x), np.array(y)).tolist()

    # assert len(x[0]) == len(y), f"Invalid dimensions: {len(x[0])} != {len(y)}"

    # result = [[0 for _ in range(len(y[0]))] for _ in range(len(x))]

    # for i in range(len(x)):
    #     for j in range(len(y[0])):
    #         for k in range(len(y)):
    #             result[i][j] += x[i][k] * y[k][j]

    # return result

def mul(x: list, y: float) -> list:
    # For testing purposes as numpy is faster
    return np.multiply(np.array(x), y).tolist()

    # return [[value * y for value in row] for row in x]

def transpose(x: list) -> list:
    # For testing purposes as numpy is faster
    return np.transpose(np.array(x)).tolist()

    # return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]

def randn(x: int, y: int) -> list:
    # For testing purposes as numpy is faster
    return np.random.randn(x, y).tolist()

    # return [[random.uniform(-1, 1) for _ in range(y)] for _ in range(x)]

def zeros(x: int, y: int) -> list:
    # For testing purposes as numpy is faster
    return np.zeros((x, y)).tolist()

    # return [[0 for _ in range(y)] for _ in range(x)]