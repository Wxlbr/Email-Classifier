import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperbolic tangent function
def tanh(x):
    return np.tanh(x)

np.random.seed(42)

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Weight initialization using Xavier initialization
        limit = 1 / np.sqrt(hidden_size)
        self.Wr = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.Ur = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.br = np.zeros((hidden_size, 1))
        self.Wz = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.Uz = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.bz = np.zeros((hidden_size, 1))
        self.Wh = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.Uh = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1))

        self.Wy = np.random.uniform(-limit, limit, (1, hidden_size))
        self.by = np.zeros((1, 1))

        # Adam optimizer
        self.m = [np.zeros_like(param) for param in [self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh]]
        self.v = [np.zeros_like(param) for param in [self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh]]

    def forward(self, x, h_prev):
        z = sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h_prev) + self.bz)
        r = sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h_prev) + self.br)
        h_hat = tanh(np.dot(self.Wh, x) + np.dot(self.Uh, np.multiply(r, h_prev)) + self.bh)
        h_next = np.multiply((1 - z), h_prev) + np.multiply(z, h_hat)
        y = sigmoid(np.dot(self.Wy, h_next) + self.by)
        return y, h_next

    def backward(self, x, h_prev, dh_next):
        z = sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h_prev) + self.bz)
        r = sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h_prev) + self.br)
        h_hat = tanh(np.dot(self.Wh, x) + np.dot(self.Uh, np.multiply(r, h_prev)) + self.bh)

        dh_hat = dh_next * z
        dtanh = (1 - h_hat ** 2) * dh_hat

        dbh = dtanh
        dWh = np.dot(dtanh, x.T)
        dUh = np.dot(dtanh, np.multiply(r, h_prev).T)
        dh_prev1 = np.dot(self.Uh.T, np.multiply(r, dtanh))
        dr = np.dot(self.Uh.T, np.multiply(h_prev, dtanh))
        dz = dh_next * (h_prev - h_hat)

        dWr = np.dot(r * (1 - r) * dr, x.T)
        dUr = np.dot(r * (1 - r) * dr, h_prev.T)
        dbr = r * (1 - r) * dr

        dWz = np.dot(z * (1 - z) * dz, x.T)
        dUz = np.dot(z * (1 - z) * dz, h_prev.T)
        dbz = z * (1 - z) * dz

        dx = np.dot(self.Wz.T, dWz) + np.dot(self.Wr.T, dWr) + np.dot(self.Wh.T, dWh)
        dh_prev2 = np.dot(self.Uz.T, dUz) + np.dot(self.Ur.T, dUr) + np.dot(self.Uh.T, dUh)

        dinputs = dx
        dparams = [dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh]

        # Clip gradients to mitigate exploding gradients
        for dparam in dparams:
            np.clip(dparam, -1, 1, out=dparam)

        return dinputs, dh_prev1 + dh_prev2, dparams

    def update_params(self, dparams, learning_rate):
        for param, dparam in zip([self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh], dparams):
            param -= learning_rate * dparam

        return

        # Adam optimizer
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        for param, dparam, m, v in zip([self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh], dparams, self.m, self.v):
            m = beta1 * m + (1 - beta1) * dparam
            v = beta2 * v + (1 - beta2) * (dparam ** 2)
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def predict(self, x, h_prev):
        y_pred, h_next = self.forward(x, h_prev)
        return y_pred

    def train(self, x, h_prev, y, learning_rate=0.01):
        y_pred, h_next = self.forward(x, h_prev)
        loss = np.mean((y_pred - y) ** 2)  # Adjusted the loss calculation

        dloss = 2 * (y_pred - y)

        dinputs, dh_prev, dparams = self.backward(x, h_prev, dloss)

        self.update_params(dparams, learning_rate)

        return loss, h_next

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
y = y.reshape(-1, 1)  # Reshape y to make it compatible with the model

# Normalize the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the GRU model
input_size = X.shape[1]
hidden_size = 300

gru_cell = GRUCell(input_size, hidden_size)

# Training the model
epochs = 150
learning_rate = 0.001

for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i].reshape(-1, 1)  # Reshape the input for the GRU cell
        h_prev = np.random.randn(hidden_size, 1)
        label = y_train[i]
        loss, h_prev = gru_cell.train(x, h_prev, label, learning_rate)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

# Testing the model
correct = 0
for i in range(len(X_test)):
    x = X_test[i].reshape(-1, 1)  # Reshape the input for the GRU cell
    h_prev = np.random.randn(hidden_size, 1)
    true_label = y_test[i][0]
    predicted_label = gru_cell.predict(x, h_prev).mean()  # Adjusted the predicted label calculation
    print(f"Predicted: {predicted_label}, True: {true_label}")
    if true_label == (predicted_label > 0.5):
        correct += 1

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy}")