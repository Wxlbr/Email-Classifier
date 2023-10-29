import numpy as np

class Matrix:
    def __init__(self, values):
        self.values = values
        self.rows = len(values)
        self.cols = len(values[0])

    def __getitem__(self, index):
        return self.values[index]

    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.cols == other.rows, f"Invalid dimensions: {self.cols} != {other.rows}"

            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    total = 0
                    for k in range(self.cols):
                        total += self.values[i][k] * other[k][j]
                    row.append(total)
                result.append(row)

            return Matrix(result)

        if isinstance(other, (int, float)):
            return Matrix([[value * other for value in row] for row in self.values])

        raise TypeError(f"Invalid type for operation addition: {type(other)}")

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.rows == other.rows and self.cols == other.cols, f"Invalid dimensions: {self.rows}x{self.cols} != {other.rows}x{other.cols}"

            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self.values[i][j] + other[i][j])
                result.append(row)

            return Matrix(result)

        raise TypeError(f"Invalid type for operation addition: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.rows == other.rows and self.cols == other.cols, f"Invalid dimensions: {self.rows}x{self.cols} != {other.rows}x{other.cols}"

            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self.values[i][j] - other[i][j])
                result.append(row)

            return Matrix(result)

        raise TypeError(f"Invalid type for operation addition: {type(other)}")

    def __repr__(self):
        return f"Matrix({self.values})"

    def to_list(self):
        return self.values

def dot(x: list, y: list) -> list:
    # For testing purposes as numpy is faster
    return np.dot(x, y)

    assert len(x[0]) == len(y), f"Invalid dimensions: {len(x[0])} != {len(y)}"

    result = []
    for i in range(len(x)):
        row = []
        for j in range(len(y[0])):
            total = 0
            for k in range(len(y)):
                total += x[i][k] * y[k][j]
            row.append(total)
        result.append(row)

    return result

def mul(x: list, y: float) -> list:
    # For testing purposes as numpy is faster
    return np.multiply(x, y)

    return [[value * y for value in row] for row in x]

if __name__ == '__main__':
    # Test matrix multiplication
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    print(a * b)
    print(np.dot(a.to_list(), b.to_list()))

    # Test matrix addition
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    print(a + b)
    print(np.add(a.to_list(), b.to_list()))

    # Test matrix subtraction
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    print(a - b)
    print(np.subtract(a.to_list(), b.to_list()))
