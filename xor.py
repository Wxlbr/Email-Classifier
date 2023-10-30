import numpy as np

from dense import Dense
from activation import Tanh
from network import train, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3, activation=Tanh()),
    Dense(3, 1, activation=Tanh())
]

# train
train(network, X, Y, epochs=10000, learning_rate=0.1, validation_data=(X, Y))

# predict
for x in X:
    print(f"{x[0]} XOR {x[1]} = {predict(network, x)[0][0] > 0.5}")