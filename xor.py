import numpy as np

from dense import Dense
from activations import Tanh
from losses import mse, mse_derivative
from network import train, accuracy

from sklearn.model_selection import train_test_split

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=1000, noise=0.1)
print(X.shape, Y.shape)
X = np.reshape(X, (1000, 2, 1))
Y = np.reshape(Y, (1000, 1, 1))

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


input_size = X_train.shape[1]
hidden_size = 3
output_size = Y_train.shape[1]

network = [
    Dense(input_size, hidden_size),
    Tanh(),
    Dense(hidden_size, output_size),
    Tanh()
]

# train
train(network, mse, mse_derivative, X_train, Y_train, epochs=100, learning_rate=0.1, validation_data=(X_test, Y_test))

# accuracy
print(f"Accuracy: {accuracy(network, X_test, Y_test)}%")