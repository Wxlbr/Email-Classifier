import numpy as np
import requests
import os

from dense import Dense, Convolutional
from reshape import Reshape
from activation import Sigmoid
from network import train, predict, accuracy

def get_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url, allow_redirects=True)
        open(path, 'wb').write(r.content)

    return path

def load_data():
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = origin_folder + "mnist.npz"
    path = get_file(path, "mnist.npz")
    with np.load(path) as data:
        train_examples = data["x_train"]
        train_labels = data["y_train"]
        test_examples = data["x_test"]
        test_labels = data["y_test"]

    return (train_examples, train_labels), (test_examples, test_labels)

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, 2)
    y = y.reshape(len(y), 2, 1)
    return x, y

if __name__ == '__main__':
    # load MNIST from server, limit to 100 images per class since we're not training on GPU
    (x_train, y_train), (x_test, y_test) = load_data()

    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)


    print(x_train.shape)

    # neural network
    network = [
        Convolutional((1, 28, 28), 3, 5, activation=Sigmoid()),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100, activation=Sigmoid()),
        Dense(100, 2, activation=Sigmoid())
    ]

    # train
    train(
        network,
        x_train,
        y_train,
        epochs=20,
        learning_rate=0.1,
        loss='binary_crossentropy',
        validation_data=(x_test, y_test),
        verbose=True
    )

    # test
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

    # Accuracy
    accuracy = accuracy(network, x_test, y_test)

    print(f"Accuracy: {accuracy:.4f}%")