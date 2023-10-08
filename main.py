import numpy as np
import pandas as pd

# Activation functions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)

# GRU Layer class


class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters
        # Should have shape (hidden_size, input_size)
        self.Wz = np.random.randn(hidden_size, input_size)
        self.Uz = np.random.randn(hidden_size, hidden_size)
        self.bz = np.zeros((hidden_size, 1))

        self.Wr = np.random.randn(hidden_size, input_size)
        self.Ur = np.random.randn(hidden_size, hidden_size)
        self.br = np.zeros((hidden_size, 1))

        self.Wh = np.random.randn(hidden_size, input_size)
        self.Uh = np.random.randn(hidden_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        # Reset gate
        rz = sigmoid(np.dot(self.Wz, np.array(x).reshape(-1, 1)) +
                     np.dot(self.Uz, h_prev) + self.bz)

        # Update gate
        rr = sigmoid(np.dot(self.Wr, np.array(x).reshape(-1, 1)) +
                     np.dot(self.Ur, h_prev) + self.br)

        # Candidate hidden state
        h_candidate = tanh(np.dot(self.Wh, np.array(x).reshape(-1, 1)) +
                           np.dot(self.Uh, rr * h_prev) + self.bh)

        # New hidden state
        h = rz * h_prev + (1 - rz) * h_candidate

        return h, rz, rr, h_candidate

    def backward(self, x, h_prev, h, rz, rr, h_candidate, dh_next):
        # Gradient with respect to the output hidden state
        dh = dh_next

        # Gradient with respect to the reset gate
        drz = dh * (h_candidate - h_prev)
        drz_dz = rz * (1 - rz)
        dz = drz * drz_dz

        # Ensure that x has the correct shape (input_size, 1)
        x_reshaped = np.array(x).reshape((self.input_size, 1))

        # Gradient with respect to the update gate
        drr = np.dot(self.Uh.T, dh * (1 - rz) * (1 - h_candidate**2))
        drr_dr = rr * (1 - rr)
        dr = drr * drr_dr

        # Gradient with respect to the candidate hidden state
        dh_candidate = np.dot(self.Uh.T, dh * (1 - rz) * (1 - h_candidate**2))

        # Gradient with respect to the new hidden state
        dh_prev = np.dot(self.Uz.T, dz) + np.dot(self.Ur.T,
                                                 dr) + np.dot(self.Uh.T, dh_candidate)

        # Gradients with respect to parameters
        dWz = np.dot(dz, x_reshaped.T)
        dUz = np.dot(dz, h_prev.T)
        dbz = np.sum(dz, axis=1, keepdims=True)

        dWr = np.dot(dr, x_reshaped.T)
        dUr = np.dot(dr, h_prev.T)
        dbr = np.sum(dr, axis=1, keepdims=True)

        dWh = np.dot(dh_candidate * (1 - rz), x_reshaped.T)
        dUh = np.dot(dh_candidate * (1 - rz), (rr * h_prev).T)
        dbh = np.sum(dh_candidate * (1 - rz), axis=1, keepdims=True)

        return dh_prev, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            h_prev = np.zeros((self.hidden_size, 1))
            for x, target in zip(X, y):
                # Forward pass
                h, rz, rr, h_candidate = self.forward(x, h_prev)

                # Calculate MSE loss
                loss = np.mean((h - target)**2)
                total_loss += loss

                # Compute gradients using backpropagation
                dh_next = 2 * (h - target)
                dh_prev, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh = self.backward(
                    x, h_prev, h, rz, rr, h_candidate, dh_next)

                # Update parameters using Adam optimizer (simplified)
                # In practice, you would use a dedicated optimization library like TensorFlow or PyTorch
                # to handle parameter updates.
                learning_rate = 0.01
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8

                # Update parameter values
                self.Wz -= learning_rate * dWz
                self.Uz -= learning_rate * dUz
                self.bz -= learning_rate * dbz

                self.Wr -= learning_rate * dWr
                self.Ur -= learning_rate * dUr
                self.br -= learning_rate * dbr

                self.Wh -= learning_rate * dWh
                self.Uh -= learning_rate * dUh
                self.bh -= learning_rate * dbh

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


def mod(x):
    return x if x > 0 else -x


# Example usage
if __name__ == "__main__":
    input_size = 100
    hidden_size = 4
    gru = GRU(input_size, hidden_size)

    num_inputs = 100

    # X_train = [np.random.randn(input_size, 1) for _ in range(10)]
    # print(X_train)
    # y_train = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    with open('./inc/emails.csv', 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)
        # row
        X_train = [data.iloc[i][:input_size].values.tolist() if input_size < len(
            data.iloc[i]) else data.iloc[i][:len(
                data.iloc[i])-1].values.tolist() for i in range(num_inputs)]
        # column
        y_train = [data['Prediction'][i] for i in range(num_inputs)]

    print(X_train)
    print(y_train)

    gru.train(X_train, y_train, learning_rate=0.01, epochs=100)

    test_size = 100
    num_test = 100

    X_test = [data.iloc[i][input_size:(input_size + test_size)].values.tolist() if input_size + test_size < len(
        data.iloc[i]) else data.iloc[i][input_size:len(
            data.iloc[i])-1].values.tolist() for i in range(num_test)]

    y_test = [data['Prediction'][i+input_size] for i in range(num_test)]

    predictions = gru.predict(X_test)
    correct = 0
    positives = 0
    negatives = 0

    for i, pred in enumerate(predictions):
        # print(f"Prediction {i + 1}:")
        p = 1 if mod(pred.mean()) > 0.5 else 0
        if p == y_test[i]:
            correct += 1
        if p == 1:
            positives += 1
        else:
            negatives += 1

    print(f"Accuracy: {correct / len(y_test)}")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")
