import numpy as np
import pandas as pd
import tensorflow as tf

# Example usage
if __name__ == "__main__":
    input_size = 100
    hidden_size = 100

    model = tf.keras.Sequential([
        tf.keras.layers.GRU(hidden_size, input_shape=(input_size, 1), return_sequences=True),
        tf.keras.layers.GRU(hidden_size, return_sequences=True),
        tf.keras.layers.GRU(hidden_size, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with open('./inc/emailsHotEncoding.csv', 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)

    num_inputs = 4500
    num_tests = 500

    X_train = np.array([data.iloc[i][:input_size].values for i in range(num_inputs)])
    X_train = X_train.reshape(X_train.shape[0], input_size, 1).astype(np.float32)

    y_train = np.array([data['Prediction'][i] for i in range(num_inputs)]).astype(np.float32)

    model.fit(X_train, y_train, epochs=10)

    test_size = input_size

    X_test = np.array([data.iloc[i][input_size:(input_size + test_size)].values for i in range(num_tests)])
    X_test = X_test.reshape(X_test.shape[0], input_size, 1).astype(np.float32)

    y_test = np.array([data['Prediction'][i + input_size] for i in range(num_tests)]).astype(np.float32)

    predictions = model.predict(X_test)
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
