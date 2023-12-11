import math
import random


def dot(x: list, y: list) -> list:

    assert len(x[0]) == len(y), f"Invalid dimensions: {len(x[0])} != {len(y)}"

    result = [[0 for _ in range(len(y[0]))] for _ in range(len(x))]

    for i, row_x in enumerate(x):
        for j, _ in enumerate(y[0]):
            for k, row_y in enumerate(y):
                result[i][j] += row_x[k] * row_y[j]

    return result


def multiply(x: list, y: float) -> list:
    # print(x, y)
    return [[value * y for value in row] for row in x]


def transpose(x: list) -> list:
    return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]


def randn(x: int, y: int) -> list:
    return [[random.uniform(-1, 1) for _ in range(y)] for _ in range(x)]


def zeros(x: int, y: int) -> list:
    return [[0] * y] * x


def ones(x: int, y: int) -> list:
    return [[1] * y] * x


def exp(x: list) -> list:
    if isinstance(x, list):
        return [[math.exp(value) for value in row] for row in x]
    return math.exp(x)


def flatten(x: list) -> list:
    return [item for sublist in x for item in sublist]


def mean(x: list) -> float:
    x = flatten(x)
    return sum(x) / len(x)


def log(x: list) -> list:
    if isinstance(x, list):
        return [[math.log(value) for value in row] for row in x]
    return math.log(x)


# def size(x: list) -> int:
#     x = flatten(x)
#     return len(x)


def negative(x: list) -> list:
    return [[-value for value in row] for row in x]


def add_matrices(x: list, y: list) -> list:
    assert len(x) == len(y), f"Invalid dimensions: {len(x)} != {len(y)}"
    assert len(x[0]) == len(
        y[0]), f"Invalid dimensions: {len(x[0])} != {len(y[0])}"

    return [[x[i][j] + y[i][j] for j in range(len(x[0]))] for i in range(len(x))]


def sub_matrices(x: list, y: list) -> list:
    assert len(x) == len(y), f"Invalid dimensions: {len(x)} != {len(y)}"
    assert len(x[0]) == len(
        y[0]), f"Invalid dimensions: {len(x[0])} != {len(y[0])}"

    return [[x[i][j] - y[i][j] for j in range(len(x[0]))] for i in range(len(x))]
