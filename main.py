import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# Activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class BatchNorm:
    def __init__(self, input_size):
        self.input_size = input_size
        self.gamma = np.ones((input_size, 1))
        self.beta = np.zeros((input_size, 1))
        self.epsilon = 1e-5

    def forward(self, x):
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        output = self.gamma * x_normalized + self.beta
        return output

# GRU cell class
class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters
        # Xavier initialization
        self.Wz = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2) # Shape (hidden_size, input_size)
        self.Uz = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size / 2) # Shape (hidden_size, hidden_size)
        self.bz = np.zeros((hidden_size, 1)) # Shape (hidden_size, 1)

        self.Wr = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2) # Shape (hidden_size, input_size)
        self.Ur = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size / 2) # Shape (hidden_size, hidden_size)
        self.br = np.zeros((hidden_size, 1)) # Shape (hidden_size, 1)

        self.Wh = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2) # Shape (hidden_size, input_size)
        self.Uh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size / 2) # Shape (hidden_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))

        # Batch normalization parameters
        self.norm_z = BatchNorm(hidden_size)
        self.norm_r = BatchNorm(hidden_size)
        self.norm_h = BatchNorm(hidden_size)

        # Adam Optimizer parameters
        self.m = {}
        self.v = {}

        self.t = 0

        self.dWz = np.zeros_like(self.Wz) # Shape (hidden_size, input_size)
        self.dUz = np.zeros_like(self.Uz) # Shape (hidden_size, hidden_size)
        self.dbz = np.zeros_like(self.bz) # Shape (hidden_size, 1)

        self.dWr = np.zeros_like(self.Wr) # Shape (hidden_size, input_size)
        self.dUr = np.zeros_like(self.Ur) # Shape (hidden_size, hidden_size)
        self.dbr = np.zeros_like(self.br) # Shape (hidden_size, 1)

        self.dWh = np.zeros_like(self.Wh) # Shape (hidden_size, input_size)
        self.dUh = np.zeros_like(self.Uh) # Shape (hidden_size, hidden_size)
        self.dbh = np.zeros_like(self.bh) # Shape (hidden_size, 1)

    def forward(self, x, h_prev):
        # x = np.array(x).reshape(-1, 1)

        # Reset gate
        rz = sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h_prev) + self.bz)

        # Update gate
        rr = sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h_prev) + self.br)

        # Candidate hidden state
        h_candidate = tanh(np.dot(self.Wh, x) + np.dot(self.Uh, rr * h_prev) + self.bh)

        # Batch normalization for reset gate
        rz_normalized = self.norm_z.forward(rz)

        # Batch normalization for update gate
        rr_normalized = self.norm_r.forward(rr)

        # Batch normalization for candidate hidden state
        h_candidate_normalized = self.norm_h.forward(h_candidate)

        # New hidden state
        # h = rz * h_prev + (1 - rz) * h_candidate
        h = rz_normalized * h_prev + (1 - rz_normalized) * h_candidate_normalized

        # return h, rz, rr, h_candidate
        return h, rz_normalized, rr_normalized, h_candidate_normalized

    def backward(self, x, h_prev, rz, rr, h_candidate, dh, update=True):
        # Gradient with respect to the hidden state
        # dh = dh * (1 - rz) + np.dot(self.Uh.T, dh * (1 - rz) * (1 - h_candidate**2)) # Check this

        # Ensure that x has the correct shape (input_size, 1)
        x = np.array(x).reshape((self.input_size, 1))

        # Gradient with respect to the reset gate
        dz = (dh * (h_candidate - h_prev)) * (rz * (1 - rz))

        # Gradient with respect to the update gate
        dr = np.dot(self.Uz.T, dz) * (rr * (1 - rr))

        # Gradient with respect to the candidate hidden state
        dh_candidate = np.dot(self.Uh.T, dh * (1 - rz) * (1 - h_candidate**2))

        print(dz.shape, x.shape, h_prev.shape)

        # Gradients with respect to parameters
        self.dWz[self.t] = np.dot(dz, x)
        self.dUz[self.t] = np.dot(dz, h_prev)
        self.dbz[self.t] = np.sum(dz, axis=1, keepdims=True)

        self.dWr[self.t] = np.dot(dr, x)
        self.dUr[self.t] = np.dot(dr.T, h_prev)
        self.dbr[self.t] = np.sum(dr, axis=1, keepdims=True)

        # print(dr.T.shape, h_prev.shape)
        # print((dh_candidate * (1 - rz)).T.shape, (rr * h_prev).T.shape)
        # print((dh_candidate * (1 - rz)).T.shape, (rr * h_prev).shape)
        print((dh_candidate * (1 - rz)).shape, (rr * h_prev).T.shape)
        # print((dh_candidate * (1 - rz)).shape, (rr * h_prev).shape)

        print(rr.shape, h_prev.shape)

        self.dWh[self.t] = np.dot(dh_candidate * (1 - rz), x.T).T
        self.dUh[self.t] = np.dot(dh_candidate * (1 - rz), (rr * h_prev).T)
        self.dbh[self.t] = np.sum(dh_candidate * (1 - rz), axis=1, keepdims=True)

        if update:
            self.update()

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            h_prev = np.zeros((self.hidden_size, 1))
            self.t = 0  # Initialize time step

            for x, target in zip(X, y):
                # Forward pass
                h, rz, rr, h_candidate = self.forward(x, h_prev)

                # Calculate MSE loss
                loss = np.mean((h - target)**2)
                total_loss += loss

                # Compute gradients using backpropagation
                dh_next = 2 * (h - target)
                self.backward(x, h_prev, rz, rr, h_candidate, dh_next)

                self.t += 1  # Increase the time step

                h_prev = h

            # Print average loss for the epoch
            average_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    def predict(self, X):
        predictions = []
        h_prev = np.zeros((self.hidden_size, 1))
        for x in X:
            h, _, _, _ = self.forward(x, h_prev)
            predictions.append(h)
            h_prev = h
        return predictions

    def update(self):
        # Update parameters using Adam Optimiser
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-4
        learning_rate = 0.01

        for param in ['Wz', 'Uz', 'bz', 'Wr', 'Ur', 'br', 'Wh', 'Uh', 'bh']:
            dparam = getattr(self, 'd' + param)[self.t]
            if param not in self.m:
                self.m[param] = np.zeros_like(dparam)
                self.v[param] = np.zeros_like(dparam)

            self.m[param] = beta1 * self.m[param] + (1 - beta1) * dparam
            self.v[param] = beta2 * self.v[param] + (1 - beta2) * (dparam ** 2)

            m_correlated = self.m[param] / (1 - beta1 ** (self.t + 1))
            v_correlated = self.v[param] / (1 - beta2 ** (self.t + 1))

            # Update the parameter
            param_val = getattr(self, param)

            numerator = learning_rate * m_correlated
            denominator = np.sqrt(v_correlated) + epsilon
            difference = numerator / denominator

            if len(param_val.shape) == 2 and len(difference.T.shape) == 2:
                param_val -= difference
            else:
                param_val -= difference.T

            setattr(self, param, param_val)

        # Clip gradients to mitigate exploding gradients
        for param_name in ['Wz', 'Uz', 'bz', 'Wr', 'Ur', 'br', 'Wh', 'Uh', 'bh']:
            np.clip(self.__dict__[param_name], -1, 1, out=self.__dict__[param_name])



# Example usage
if __name__ == "__main__":
    # Read data from inc/kaggleDataset.csv
    with open('inc/kaggleDataset.csv', 'r', encoding='utf-8') as f:
        emails = pd.read_csv(f)

    # Shuffle the data
    emails = emails.sample(frac=1, random_state=42).reset_index(drop=True)

    # Get the labels
    labels = emails.pop('label')
    labels = labels.map({'ham': 0, 'spam': 1})

    # Get the text
    text = emails.pop('text')

    # Split the text into lowercase words
    text = text.apply(lambda x: x.lower().split(' '))

    # Word2Vec
    model = Word2Vec(sentences=text, vector_size=100, window=5, min_count=1, workers=30)
    words = list(model.wv.index_to_key)

    # Convert text to word vectors
    word_vectors = []
    for sentence in text:
        sentence_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(sentence_vectors) > 0:
            word_vectors.append(sentence_vectors)
        else:
            word_vectors.append([np.zeros(100)])

    # Padding the sequences to make them of equal length
    padded_vectors = tf.keras.preprocessing.sequence.pad_sequences(word_vectors, padding='post')

    # Reduce the data size by sampling to 1000 vectors of length 100 each of single words
    sample_size = 1000
    sample_indices = np.random.choice(len(padded_vectors), sample_size, replace=False)

    padded_vectors_sampled = padded_vectors[sample_indices]
    labels_sampled = labels.iloc[sample_indices]

    # Get the input size
    input_size = padded_vectors_sampled.shape[2]

    # exit()

    # Split the sampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded_vectors_sampled, labels_sampled, test_size=0.2, random_state=42)

    hidden_size = 4

    model = GRU(input_size, hidden_size)

    model.train(X_train, y_train, epochs=1)
