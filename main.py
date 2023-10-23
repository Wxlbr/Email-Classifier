import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Tanh function
def tanh(x):
    return np.tanh(x)

# Derivative of tanh function
def tanh_derivative(x):
    return 1 - x ** 2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def relu(x):
    return np.maximum(0, x)

class Word2VecWrapper:
    def __init__(self, sentences):
        self.word2vec = Word2Vec(sentences, min_count=1)

    def vectorise(self, sentence, pad):
        vecs = [self.word2vec.wv[word] for word in sentence if word in self.word2vec.wv]
        if len(vecs) < pad:
            vecs += [[0] * self.word2vec.vector_size] * (pad - len(vecs))
        return vecs[:pad]

class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.Wr = np.random.randn(hidden_size, input_size)
        self.Ur = np.random.randn(hidden_size, hidden_size)
        self.Br = np.zeros((hidden_size, 1))

        self.Wz = np.random.randn(hidden_size, input_size)
        self.Uz = np.random.randn(hidden_size, hidden_size)
        self.Bz = np.zeros((hidden_size, 1))

        self.Wh = np.random.randn(hidden_size, input_size)
        self.Uh = np.random.randn(hidden_size, hidden_size)
        self.Bh = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size)
        self.By = np.zeros((output_size, 1))

    def forward(self, x, h_prev):
        z = sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h_prev) + self.Bz)
        r = sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h_prev) + self.Br)
        h_hat = tanh(np.dot(self.Wh, x) + np.dot(self.Uh, np.multiply(r, h_prev)) + self.Bh)
        h_next = np.multiply(1 - z, h_prev) + np.multiply(z, h_hat)
        y = np.dot(self.Wy, h_next) + self.By
        return y, h_next, z, r, h_hat

    def backward(self, d_y, h_prev, h_next, x, z, r, h_hat):
        d_Wy = np.dot(d_y, h_next.T)
        d_By = d_y
        d_h_next = np.dot(self.Wy.T, d_y)

        d_z = d_h_next * (h_prev - h_hat)
        d_r = np.dot(self.Uh.T, np.multiply(d_h_next, h_prev - h_hat)) * h_prev * (1 - r)

        d_h_hat = np.dot(self.Uh.T, np.multiply(d_h_next, 1 - r))
        d_Wh = np.dot(d_h_hat, x.T)
        d_Uh = np.dot(d_h_hat, np.multiply(r, h_prev).T)
        d_Bh = d_h_hat

        d_Wz = np.dot(d_z, x.T)
        d_Uz = np.dot(d_z, h_prev.T)
        d_Bz = d_z

        d_Wr = np.dot(d_r, x.T)
        d_Ur = np.dot(d_r, h_prev.T)
        d_Br = d_r

        d_x = (np.dot(self.Wr.T, d_r) + np.dot(self.Wz.T, d_z) + np.dot(self.Wh.T, d_h_hat))

        # Clip to prevent exploding gradients
        for gradient in [d_Wy, d_By, d_Wh, d_Uh, d_Bh, d_Wz, d_Uz, d_Bz, d_Wr, d_Ur, d_Br]:
            np.clip(gradient, -1, 1, out=gradient)

        return {
            'Wy': d_Wy,
            'By': d_By,
            'Wh': d_Wh,
            'Uh': d_Uh,
            'Bh': d_Bh,
            'Wz': d_Wz,
            'Uz': d_Uz,
            'Bz': d_Bz,
            'Wr': d_Wr,
            'Ur': d_Ur,
            'Br': d_Br
        }

    def adam_optimizer(self, gradients, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-4):
        v_corrected = {}
        s_corrected = {}
        for key in gradients.keys():
            v[key] = beta1 * v[key] + (1 - beta1) * gradients[key]
            v_corrected[key] = v[key] / (1 - beta1 ** t)
            s[key] = beta2 * s[key] + (1 - beta2) * (gradients[key] ** 2)
            s_corrected[key] = s[key] / (1 - beta2 ** t)

            if hasattr(self, key):
                setattr(self, key, getattr(self, key) - learning_rate * v_corrected[key] / (np.sqrt(s_corrected[key]) + epsilon))

        return self

    def train_network(self, X, Y, iterations=10, learning_rate=0.01):
        hidden_size = self.Wy.shape[1]
        v = {
            'Wy': np.zeros_like(self.Wy),
            'Wh': np.zeros_like(self.Wh),
            'Wz': np.zeros_like(self.Wz),
            'Uz': np.zeros_like(self.Uz),
            'Uh': np.zeros_like(self.Uh),
            'Ur': np.zeros_like(self.Ur),
            'By': np.zeros_like(self.By),
            'Bh': np.zeros_like(self.Bh),
            'Bz': np.zeros_like(self.Bz),
            'Br': np.zeros_like(self.Br),
            'Wr': np.zeros_like(self.Wr),
        }
        s = v.copy()

        for i in range(1, iterations + 1):
            loss = 0
            h_prev = np.zeros((hidden_size, 1))
            for j in range(len(X)):
                x, y = X[j], Y[j]
                y_pred, h_prev, z, r, h_hat = self.forward(x, h_prev)

                # Compute the predicted output and the loss
                loss += np.square(y_pred - y).sum()

                d_y = 2 * (y_pred - y)
                gradients = self.backward(d_y, h_prev, h_prev, x, z, r, h_hat)
                self.adam_optimizer(gradients, v, s, i, learning_rate)
            if i % 1 == 0:
                print(f'Iteration {i}, Loss: {loss / len(X)}')

    def accuracy(self, X, Y):
        correct = 0
        for i in range(len(X)):
            x, y = X[i], Y[i]
            y_pred, _, _, _, _ = self.forward(x, np.zeros((self.Wy.shape[1], 1)))
            if (y_pred.mean() > 0.5 == y):
                correct += 1
            print(y_pred.mean(), y_pred.mean() > 0.5, y)
        return correct / len(X) * 100

# Load the data
data = pd.read_csv('inc/kaggleDataset.csv')
text = data['text'].tolist()

# Get the labels
labels = data.pop('label')
labels = labels.map({'ham': 0, 'spam': 1})

model = Word2VecWrapper([sentence.split() for sentence in text])

# Convert the text to word vectors
padded_vectors = np.array([model.vectorise(sentence, pad=100) for sentence in text])

# Reduce the data size by sampling to 1000 vectors of length 100 each of single words
sample_size = 2000
sample_indices = np.random.choice(len(padded_vectors), sample_size, replace=False)

padded_vectors_sampled = padded_vectors[sample_indices]
labels_sampled = labels[sample_indices]

# print(padded_vectors_sampled)
print('[Gathered input data]')

# Split the sampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_vectors_sampled, labels_sampled, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Train the network
gru = GRU(100, 100, 1)
gru.train_network(X_train, y_train, iterations=20, learning_rate=0.001)

# Test the network
print(f'Accuracy: {gru.accuracy(X_test, y_test):.2f}%')