import pandas as pd
import numpy as np

from network import train, accuracy
from layer import Recurrent
from split import train_test_split


# Read data from inc/kaggleDataset.csv
with open('inc/emailsHotEncoding.csv', 'r', encoding='utf-8') as f:
    emails = pd.read_csv(f)

X = emails.drop('Prediction', axis=1).values
Y = emails['Prediction'].values
X = X.reshape((X.shape[0], X.shape[1], 1))
Y = Y.reshape((Y.shape[0], 1, 1))
# X = X.reshape((X.shape[0], 1, X.shape[1], 1))
# Y = Y.reshape((Y.shape[0], 1, 1))

max_height = X.shape[1]
max_width = 1

# print(X.shape)
# print(Y.shape)

# exit()

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Convert to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

network = [
    Recurrent(max_height, max_width)
]

# train
train(network, X_train, Y_train, 100, 0.5, validation_data=(X_test, Y_test))
train(network, X_train, Y_train, 100, 0.1, validation_data=(X_test, Y_test))
train(network, X_train, Y_train, 200, 0.05, validation_data=(X_test, Y_test))

# accuracy
print(f"Accuracy: {accuracy(network, X_test, Y_test):.4f}%")