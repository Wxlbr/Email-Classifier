import numpy as np

def dot(x: list, y: list) -> list:
    # For testing purposes as numpy is faster
    return np.dot(x, y)

    assert len(x[0]) == len(y), f"Invalid dimensions: {len(x[0])} != {len(y)}"

    result = [[0 for _ in range(len(y[0]))] for _ in range(len(x))]

    for i in range(len(x)):
        for j in range(len(y[0])):
            for k in range(len(y)):
                result[i][j] += x[i][k] * y[k][j]

    return result

def mul(x: list, y: float) -> list:
    # For testing purposes as numpy is faster
    return np.multiply(x, y)

    return [[value * y for value in row] for row in x]