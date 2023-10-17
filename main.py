import numpy as np
import pandas as pd

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
        # Xavier initialisation
        self.Wz = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2)
        self.Uz = np.eye(hidden_size) * 0.01
        self.bz = np.zeros((hidden_size, 1)) / np.sqrt(hidden_size / 2)

        self.Wr = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2)
        self.Ur = np.eye(hidden_size) * 0.01
        self.br = np.zeros((hidden_size, 1)) / np.sqrt(hidden_size / 2)

        self.Wh = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2)
        self.Uh = np.eye(hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1)) / np.sqrt(hidden_size / 2)

        # Batch normalization parameters
        self.norm_z = BatchNorm(hidden_size)
        self.norm_r = BatchNorm(hidden_size)
        self.norm_h = BatchNorm(hidden_size)

        # Adam Optimiser parameters
        self.m = {}
        self.v = {}

        self.t = 0

        self.dWz = np.zeros((input_size, hidden_size, input_size))
        self.dUz = np.zeros((input_size, hidden_size, hidden_size))
        self.dbz = np.zeros((input_size, hidden_size, 1))

        self.dWr = np.zeros((input_size, hidden_size, input_size))
        self.dUr = np.zeros((input_size, hidden_size, hidden_size))
        self.dbr = np.zeros((input_size, hidden_size, 1))

        self.dWh = np.zeros((input_size, hidden_size, input_size))
        self.dUh = np.zeros((input_size, hidden_size, hidden_size))
        self.dbh = np.zeros((input_size, hidden_size, 1))

    def forward(self, x, h_prev):
        # Reset gate
        rz = sigmoid(np.dot(self.Wz, np.array(x).reshape(-1, 1)) + np.dot(self.Uz, h_prev) + self.bz)

        # Update gate
        rr = sigmoid(np.dot(self.Wr, np.array(x).reshape(-1, 1)) + np.dot(self.Ur, h_prev) + self.br)

        # Candidate hidden state
        h_candidate = tanh(np.dot(self.Wh, np.array(x).reshape(-1, 1)) + np.dot(self.Uh, rr * h_prev) + self.bh)

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
        x_reshaped = np.array(x).reshape((self.input_size, 1))

        # Gradient with respect to the reset gate
        dz = (dh * (h_candidate - h_prev)) * (rz * (1 - rz))

        # Gradient with respect to the update gate
        dr = (np.dot(self.Uh.T, dh * (1 - rz) * (1 - h_candidate**2))) * (rr * (1 - rr))

        # Gradient with respect to the candidate hidden state
        dh_candidate = np.dot(self.Uh.T, dh * (1 - rz) * (1 - h_candidate**2))

        # Gradients with respect to parameters
        self.dWz[self.t] = np.dot(dz, x_reshaped.T)
        self.dUz[self.t] = np.dot(dz, h_prev.T)
        self.dbz[self.t] = np.sum(dz, axis=1, keepdims=True)

        self.dWr[self.t] = np.dot(dr, x_reshaped.T)
        self.dUr[self.t] = np.dot(dr, h_prev.T)
        self.dbr[self.t] = np.sum(dr, axis=1, keepdims=True)

        self.dWh[self.t] = np.dot(dh_candidate * (1 - rz), x_reshaped.T)
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

            setattr(self, param, getattr(self, param) - learning_rate * m_correlated / (np.sqrt(v_correlated) + epsilon))

        # Clip gradients to mitigate exploding gradients
        for param_name in ['Wz', 'Uz', 'bz', 'Wr', 'Ur', 'br', 'Wh', 'Uh', 'bh']:
            np.clip(self.__dict__[param_name], -1, 1, out=self.__dict__[param_name])

# Example usage
if __name__ == "__main__":
    input_size = 100
    hidden_size = 4
    gru = GRU(input_size, hidden_size)

    with open('./inc/emails.csv', 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)

    # num_inputs = round(len(data) * 0.8)
    # num_tests = round(len(data) * 0.2)

    num_inputs = 100
    num_tests = 100

    X_train = [data.iloc[i][:input_size].values.tolist() if input_size < len(
        data.iloc[i]) else data.iloc[i][:len(
            data.iloc[i])-1].values.tolist() for i in range(num_inputs)]

    y_train = [data['Prediction'][i] for i in range(num_inputs)]

    # print(X_train)
    # print(y_train)

    gru.train(X_train, y_train, epochs=100)

    test_size = input_size

    X_test = [data.iloc[i][input_size:(input_size + test_size)].values.tolist() if input_size + test_size < len(
        data.iloc[i]) else data.iloc[i][input_size:len(
            data.iloc[i])-1].values.tolist() for i in range(num_tests)]

    y_test = [data['Prediction'][i+input_size] for i in range(num_tests)]

    predictions = gru.predict(X_test)
    correct = 0
    positives = 0
    negatives = 0

    for i, pred in enumerate(predictions):
        # print(f"Prediction {i + 1}:")
        print(f"Predicted: {pred.mean()}")
        p = 1 if pred.mean() > 0.5 else 0
        if p == y_test[i]:
            correct += 1
        if p == 1:
            positives += 1
        else:
            negatives += 1

    print(f"Accuracy: {correct / len(y_test)}")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")