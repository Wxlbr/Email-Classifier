def train_test_split(X, Y, test_size=0.2):
    assert len(X) == len(Y) and len(X) > 0

    train_size = int(len(X) * (1 - test_size))
    test_size = len(X) - train_size

    assert train_size > 0 and test_size > 0

    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    return X_train, X_test, Y_train, Y_test