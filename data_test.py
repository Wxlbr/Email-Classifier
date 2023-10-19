import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Read data from inc/kaggleDataset.csv
with open('inc/kaggleDataset.csv', 'r', encoding='utf-8') as f:
    emails = pd.read_csv(f)

# Shuffle the data
emails = emails.sample(frac=1, random_state=42).reset_index(drop=True)

# Get the labels
labels = emails.pop('label')
labels = labels.map({'ham': 0, 'spam': 1})

# Get the text
text = emails.pop('text')

# Split the text into lowercase words
text = text.apply(lambda x: x.lower().split(' '))

def djb2(text):
    h = 5381
    for char in text:
        h = ((h << 5) + h) + ord(char)
    return h % 1000

def djb2ls(ls):
    return [djb2(text) for text in ls]

# Hash the text data
text = text.apply(djb2ls)

# Pad sequences to make them of equal length
max_sequence_length = max(len(seq) for seq in text)
input_size = max_sequence_length

text = np.array([np.pad(seq, (0, max_sequence_length - len(seq))) for seq in text])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42)

hidden_size = 100

model = tf.keras.Sequential([
    tf.keras.layers.GRU(hidden_size, input_shape=(input_size, 1), return_sequences=True),
    tf.keras.layers.GRU(hidden_size, return_sequences=True),
    tf.keras.layers.GRU(hidden_size, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

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

# print(f"Accuracy: {correct / len(y_test)}")
# print(f"Positives: {positives}")
# print(f"Negatives: {negatives}")
