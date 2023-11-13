def train_test_split(X, Y, test_size=0.2):

    assert len(X) == len(Y) and len(X) > 0

    train_size = int(len(X) * (1 - test_size))
    test_size = len(X) - train_size

    assert train_size > 0 and test_size > 0

    X_train, X_test, Y_train, Y_test = [], [], [], []

    for i in range(train_size):
        X_train.append(X[i])
        Y_train.append(Y[i])

    for i in range(train_size, len(X)):
        X_test.append(X[i])
        Y_test.append(Y[i])

    return X_train, X_test, Y_train, Y_test