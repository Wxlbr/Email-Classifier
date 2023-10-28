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

if __name__ == '__main__':

    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Y = [1, 2, 3]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print(X_train)
    print(X_test)
    print(Y_train)
    print(Y_test)