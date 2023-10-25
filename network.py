def predict(network, input_value):
    output = input_value
    for layer in network:
        output = layer.forward(output)
    return output

def accuracy(network, x_test, y_test):
    correct = 0
    for i in range(x_test.shape[0]):
        x = x_test[i]
        y = y_test[i]
        pred = predict(network, x)[0][0] > 0.5

        if y[0][0] == pred:
            correct += 1

    return correct / x_test.shape[0] * 100

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, validation_data=None, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            gradient = loss_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}", end="")

            if validation_data:
                x_val, y_val = validation_data
                print(f", val_accuracy={accuracy(network, x_val, y_val)}%", end="")

            print()
