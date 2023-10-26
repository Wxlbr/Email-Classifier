from loss import MSE, BinaryCrossEntropy

def predict(network, input_value):
    output = input_value
    for layer in network:
        output = layer.forward(output)
    return output

def accuracy(network, x_test, y_test):
    correct = 0
    for x, y in zip(x_test, y_test):
        pred = predict(network, x)[0][0] > 0.5
        if y[0][0] == pred:
            correct += 1

    return correct / x_test.shape[0] * 100

def train(network, x_train, y_train, epochs=1000, learning_rate=0.01, loss='mse', validation_data=None, verbose=True):

    # select loss function
    if loss == 'mse':
        loss = MSE()
    elif loss == 'binary_crossentropy':
        loss = BinaryCrossEntropy()
    else:
        raise Exception('Unknown loss function.')

    # training loop
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss.calc(y, output)

            # backward
            gradient = loss.derivative(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        if verbose:
            print(f"{e + 1}/{epochs}, error={error / len(x_train):.4f}", end="")

            if validation_data:
                x_val, y_val = validation_data
                print(f", val_accuracy={accuracy(network, x_val, y_val):.4f}%", end="")

            print()
