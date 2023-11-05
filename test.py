from dense import Recurrent
from network import train, accuracy

import numpy as np

# XOR
input_sequence = np.array([
    [[0], [0], [0]],
    [[0], [1], [1]],
    [[1], [0], [1]],
    [[1], [1], [0]],
])

output_sequence = np.array([
    [[0]],
    [[1]],
    [[1]],
    [[0]],
])

# Define network
network = [
    Recurrent(3, 3),
    Recurrent(3, 3),
]

# Train network
train(network, input_sequence, output_sequence, epochs=1000, learning_rate=0.1, validation_data=(input_sequence, output_sequence))
