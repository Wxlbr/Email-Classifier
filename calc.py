import math
import random


def zeros(rows: int, cols: int) -> list:
    return [[0] * cols] * rows


def ones(rows: int, cols: int) -> list:
    return [[1] * cols] * rows


def dot(x: list, y: list) -> list:

    assert len(x[0]) == len(y), f"Invalid dimensions: {len(x[0])} != {len(y)}"

    result = zeros(len(x), len(y[0]))

    for i, row_x in enumerate(x):
        for j, _ in enumerate(y[0]):
            for k, row_y in enumerate(y):
                result[i][j] += row_x[k] * row_y[j]

    return result


def multiply(matrix: list, multiplier: float) -> list:
    return [[element * multiplier for element in row] for row in matrix]


def transpose(matrix: list) -> list:
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]


def randn(x: int, y: int) -> list:
    return [[random.uniform(-1, 1) for _ in range(y)] for _ in range(x)]


def exp(x):
    if isinstance(x, list):
        return [[math.exp(element) for element in row] for row in x]
    return math.exp(x)


def flatten(matrix: list) -> list:
    return [element for sublist in matrix for element in sublist]


def mean(x: list) -> float:
    # If not already flat, flatten
    if isinstance(x[0], list):
        x = flatten(x)
    return sum(x) / len(x)


def log(x: list) -> list:
    if isinstance(x, list):
        return [[math.log(element) for element in row] for row in x]
    return math.log(x)


def negative(matrix: list) -> list:
    return [[-element for element in row] for row in matrix]


def shape(x: list) -> tuple:
    if isinstance(x, list):
        return (len(x),) + shape(x[0]) if x else (len(x),)
    return (1,)


def same_dimensions(mat_1: list, mat_2: list) -> bool:
    return len(mat_1) == len(mat_2) and len(mat_1[0]) == len(mat_2[0])


def add_matrices(mat_1: list, mat_2: list) -> list:
    assert same_dimensions(
        mat_1, mat_2), f"Invalid dimensions: {shape(mat_1)} != {shape(mat_2)}"

    return [[mat_1[row][col] + mat_2[row][col] for col in range(len(mat_1[0]))] for row in range(len(mat_1))]


def sub_matrices(mat_1: list, mat_2: list) -> list:
    assert same_dimensions(
        mat_1, mat_2), f"Invalid dimensions: {shape(mat_1)} != {shape(mat_2)}"

    return [[mat_1[row][col] - mat_2[row][col] for col in range(len(mat_1[0]))] for row in range(len(mat_1))]


def multiply_matrices(mat_1: list, mat_2: list) -> list:
    assert same_dimensions(
        mat_1, mat_2), f"Invalid dimensions: {shape(mat_1)} != {shape(mat_2)}"

    return [[mat_1[row][col] * mat_2[row][col] for col in range(len(mat_1[0]))] for row in range(len(mat_1))]


def convert_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m {int(seconds % 60)}s"
    return f"{int(seconds / 3600)}h {int(seconds % 3600 / 60)}m"
