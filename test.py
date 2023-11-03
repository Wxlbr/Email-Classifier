from dense import Recurrent
from network import train, accuracy

import numpy as np

# Define a sample input sequence
input_sequence = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

# Define a sample output sequence
output_sequence = np.array([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])

# Define network
network = [
    Recurrent(3, 3),
    Recurrent(3, 3),
]

# Train network
train(network, input_sequence, output_sequence, epochs=1000, learning_rate=0.1, validation_data=(input_sequence, output_sequence))
