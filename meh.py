import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from neuralforecast.models import GRU

# Example usage
if __name__ == "__main__":
    hidden_size = 100

    model = tf.keras.Sequential([
        tf.keras.layers.GRU(hidden_size, input_shape=(None, 1), return_sequences=True),
        tf.keras.layers.GRU(hidden_size, return_sequences=True),
        tf.keras.layers.GRU(hidden_size, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with open('./inc/emailsHotEncoding.csv', 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)

    data = data.dropna()  # Drop rows with missing values
    max_sequence_length = len(data.columns) - 1  # Assume the last column is the target column

    # print(max_sequence_length)
    # exit()

    X = data.iloc[:, :max_sequence_length].values.reshape(-1, max_sequence_length, 1).astype(np.float32)
    y = data['Prediction'].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))

    # predictions = model.predict(X_test)

    # correct = 0
    # positives = 0
    # negatives = 0

    # for i, pred in enumerate(predictions):
    #     # print(f"Prediction {i + 1}:")
    #     print(f"Predicted: {pred.mean()}")
    #     if pred.mean() > 0.5:
    #         p = 1
    #         positives += 1
    #     else:
    #         p = 0
    #         negatives += 1

    #     if p == y_test[i]:
    #         correct += 1

    # accuracy = correct / len(y_test)
    # print(f"Accuracy: {accuracy}")