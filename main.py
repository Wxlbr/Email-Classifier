import numpy as np
import pandas as pd

# Activation functions

# Batch normalization function
def batch_normalize(x, gamma, beta, epsilon=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_normalized = (x - mean) / np.sqrt(var + epsilon)
    out = gamma * x_normalized + beta
    return out, x_normalized, mean, var

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# GRU Layer class
class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters
        # Xavier initialisation
        self.Wz = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2)
        # self.Uz = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size / 2)
        self.Uz = np.eye(hidden_size) * 0.01
        self.bz = np.zeros((hidden_size, 1)) / np.sqrt(hidden_size / 2)

        self.Wr = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2)
        # self.Ur = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size / 2)
        self.Ur = np.eye(hidden_size) * 0.01
        self.br = np.zeros((hidden_size, 1)) / np.sqrt(hidden_size / 2)

        self.Wh = np.random.randn(hidden_size, input_size) / np.sqrt(input_size / 2)
        # self.Uh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size / 2)
        self.Uh = np.eye(hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1)) / np.sqrt(hidden_size / 2)

        # Batch normalization parameters
        self.gamma_z = np.ones((hidden_size, 1))
        self.beta_z = np.zeros((hidden_size, 1))
        self.gamma_r = np.ones((hidden_size, 1))
        self.beta_r = np.zeros((hidden_size, 1))
        self.gamma_h_candidate = np.ones((hidden_size, 1))
        self.beta_h_candidate = np.zeros((hidden_size, 1))

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

        # Batch normalization for reset gate
        rz_normalized, _, _, _ = batch_normalize(rz, self.gamma_z, self.beta_z)

        # Batch normalization for update gate
        rr_normalized, _, _, _ = batch_normalize(rr, self.gamma_r, self.beta_r)

        # Batch normalization for candidate hidden state
        h_candidate_normalized, _, _, _ = batch_normalize(h_candidate, self.gamma_h_candidate, self.beta_h_candidate)

        # New hidden state
        # h = rz * h_prev + (1 - rz) * h_candidate
        h = rz_normalized * h_prev + (1 - rz_normalized) * h_candidate_normalized

        # return h, rz, rr, h_candidate
        return h, rz_normalized, rr_normalized, h_candidate_normalized

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

        # print('Gradient with respect to the output hidden state: ', dh)
        # print('Gradient with respect to the reset gate: ', drz)
        # print('Gradient with respect to the update gate: ', drr)
        # print('Gradient with respect to the candidate hidden state: ', dh_candidate)
        # print('Gradient with respect to the new hidden state: ', dh_prev)
        # print('Gradients with respect to parameters:')
        # print('dWz: ', dWz)
        # print('dUz: ', dUz)
        # print('dbz: ', dbz)
        # print('dWr: ', dWr)
        # print('dUr: ', dUr)
        # print('dbr: ', dbr)
        # print('dWh: ', dWh)
        # print('dUh: ', dUh)
        # print('dbh: ', dbh)

        return dh_prev, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            h_prev = np.zeros((self.hidden_size, 1))
            m = {}  # Initialize first moment estimates
            v = {}  # Initialize second moment estimates
            t = 0  # Initialize time step
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-4

            for param_name in ['Wz', 'Uz', 'bz', 'Wr', 'Ur', 'br', 'Wh', 'Uh', 'bh']:
                m[param_name] = np.zeros_like(self.__dict__[param_name])
                v[param_name] = np.zeros_like(self.__dict__[param_name])

            for x, target in zip(X, y):
                # Forward pass
                h, rz, rr, h_candidate = self.forward(x, h_prev)

                # Calculate MSE loss
                loss = np.mean((h - target)**2)
                total_loss += loss

                # print(h, rz, rr, h_candidate, loss)

                # Compute gradients using backpropagation
                dh_next = 2 * (h - target)
                dh_prev, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh = self.backward(
                    x, h_prev, h, rz, rr, h_candidate, dh_next)

                t += 1  # Increase the time step

                # Update parameters using Adam Optimiser

                # dWz
                m['Wz'] = beta1 * m['Wz'] + (1 - beta1) * dWz
                v['Wz'] = beta2 * v['Wz'] + (1 - beta2) * (dWz**2)
                m_hat = m['Wz'] / (1 - beta1**t)
                v_hat = v['Wz'] / (1 - beta2**t)
                self.Wz -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dUz
                m['Uz'] = beta1 * m['Uz'] + (1 - beta1) * dUz
                v['Uz'] = beta2 * v['Uz'] + (1 - beta2) * (dUz**2)
                m_hat = m['Uz'] / (1 - beta1**t)
                v_hat = v['Uz'] / (1 - beta2**t)
                self.Uz -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dbz
                m['bz'] = beta1 * m['bz'] + (1 - beta1) * dbz
                v['bz'] = beta2 * v['bz'] + (1 - beta2) * (dbz**2)
                m_hat = m['bz'] / (1 - beta1**t)
                v_hat = v['bz'] / (1 - beta2**t)
                self.bz -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dWr
                m['Wr'] = beta1 * m['Wr'] + (1 - beta1) * dWr
                v['Wr'] = beta2 * v['Wr'] + (1 - beta2) * (dWr**2)
                m_hat = m['Wr'] / (1 - beta1**t)
                v_hat = v['Wr'] / (1 - beta2**t)
                self.Wr -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dUr
                m['Ur'] = beta1 * m['Ur'] + (1 - beta1) * dUr
                v['Ur'] = beta2 * v['Ur'] + (1 - beta2) * (dUr**2)
                m_hat = m['Ur'] / (1 - beta1**t)
                v_hat = v['Ur'] / (1 - beta2**t)
                self.Ur -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dbr
                m['br'] = beta1 * m['br'] + (1 - beta1) * dbr
                v['br'] = beta2 * v['br'] + (1 - beta2) * (dbr**2)
                m_hat = m['br'] / (1 - beta1**t)
                v_hat = v['br'] / (1 - beta2**t)
                self.br -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dWh
                m['Wh'] = beta1 * m['Wh'] + (1 - beta1) * dWh
                v['Wh'] = beta2 * v['Wh'] + (1 - beta2) * (dWh**2)
                m_hat = m['Wh'] / (1 - beta1**t)
                v_hat = v['Wh'] / (1 - beta2**t)
                self.Wh -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dUh
                m['Uh'] = beta1 * m['Uh'] + (1 - beta1) * dUh
                v['Uh'] = beta2 * v['Uh'] + (1 - beta2) * (dUh**2)
                m_hat = m['Uh'] / (1 - beta1**t)
                v_hat = v['Uh'] / (1 - beta2**t)
                self.Uh -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # dbh
                m['bh'] = beta1 * m['bh'] + (1 - beta1) * dbh
                v['bh'] = beta2 * v['bh'] + (1 - beta2) * (dbh**2)
                m_hat = m['bh'] / (1 - beta1**t)
                v_hat = v['bh'] / (1 - beta2**t)
                self.bh -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # Clip gradients to mitigate exploding gradients
                for param_name in ['Wz', 'Uz', 'bz', 'Wr', 'Ur', 'br', 'Wh', 'Uh', 'bh']:
                    np.clip(self.__dict__[param_name], -1, 1, out=self.__dict__[param_name])

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

    gru.train(X_train, y_train, learning_rate=0.01, epochs=100)

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
