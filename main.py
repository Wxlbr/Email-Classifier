from reshape import Reshape
from network import train, accuracy
from dense import Dense, Convolutional

from sklearn.model_selection import train_test_split

import pandas as pd


# Read data from inc/kaggleDataset.csv
with open('inc/emailsHotEncoding.csv', 'r', encoding='utf-8') as f:
    emails = pd.read_csv(f)

X = emails.drop('Prediction', axis=1).values
Y = emails['Prediction'].values
X = X.reshape((X.shape[0], 1, X.shape[1], 1))
Y = Y.reshape((Y.shape[0], 1, 1))

max_height = X.shape[2]
max_width = 1

print(X.shape)
print(Y.shape)

# exit()

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

kernal_size = 1
depth = 5

network = [
    Convolutional((1, max_height, max_width), kernal_size, depth),
    Reshape((depth, max_height - kernal_size + 1, max_width - kernal_size + 1), (depth * (max_height - kernal_size + 1) * (max_width - kernal_size + 1), 1)),
    Dense(depth * (max_height - kernal_size + 1) * (max_width - kernal_size + 1), 100),
    Dense(100, 1),
]

# train
train(network, X_train, Y_train, 100, 0.01, validation_data=(X_test, Y_test))

# accuracy
print(f"Accuracy: {accuracy(network, X_test, Y_test):.4f}%")