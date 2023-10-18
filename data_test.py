import numpy as np
import pandas as pd
import tensorflow as tf

# Read data from inc/kaggleDataset.csv
with open('inc/kaggleDataset.csv', 'r', encoding='utf-8') as f:
    emails = pd.read_csv(f)

# Train-test split
train = emails.sample(frac=0.8, random_state=42)
test = emails.drop(train.index)

# Get the labels
train_labels = train.pop('label')
test_labels = test.pop('label')

# Get the text
train_text = train.pop('text')
test_text = test.pop('text')

# Map labels to numeric values (e.g., 'ham' to 0, 'spam' to 1)
train_labels = train_labels.map({'ham': 0, 'spam': 1})
test_labels = test_labels.map({'ham': 0, 'spam': 1})

# Split the text into lowercase words
train_text_temp = train_text.apply(lambda x: x.lower().split(' '))
test_text_temp = test_text.apply(lambda x: x.lower().split(' '))


def djb2(text):
    h = 5381
    for char in text:
        h = ((h << 5) + h) + ord(char)
    return h % 1000


def djb2ls(ls):
    return [djb2(text) for text in ls]


# Hash the text data
train_text = train_text_temp.apply(djb2ls)
test_text = test_text_temp.apply(djb2ls)

# Convert the text data and labels to NumPy arrays with dtype=object
train_text = np.array(train_text.tolist(), dtype=object)
test_text = np.array(test_text.tolist(), dtype=object)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Pad sequences to make them of equal length
max_sequence_length = max(len(seq) for seq in np.concatenate(
    (train_text, test_text), axis=0))
train_text = np.array(
    [np.pad(seq, (0, max_sequence_length - len(seq))) for seq in train_text])
test_text = np.array(
    [np.pad(seq, (0, max_sequence_length - len(seq))) for seq in test_text])

# Save the data to csv
# np.savetxt('inc/trainTextD.csv', train_text, delimiter=',', fmt='%s')
# np.savetxt('inc/testTextD.csv', test_text, delimiter=',', fmt='%s')
# np.savetxt('inc/trainTargetD.csv', train_labels, delimiter=',', fmt='%s')
# np.savetxt('inc/testTargetD.csv', test_labels, delimiter=',', fmt='%s')

input_size = 100 # max_sequence_length
hidden_size = 100

X_train = np.array([train_text[i][:input_size].tolist() for i in range(len(train_text))])
X_train = X_train.reshape(X_train.shape[0], input_size, 1).astype(np.float32)

y_train = np.array([train_labels[i] for i in range(len(train_labels))]).astype(np.float32)

X_test = np.array([test_text[i][:input_size].tolist() for i in range(len(test_text))])
X_test = X_test.reshape(X_test.shape[0], input_size, 1).astype(np.float32)

y_test = np.array([test_labels[i] for i in range(len(test_labels))]).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.GRU(hidden_size, input_shape=(input_size, 1), return_sequences=True),
    tf.keras.layers.GRU(hidden_size, return_sequences=True),
    tf.keras.layers.GRU(hidden_size, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

predictions = model.predict(X_test)
correct = 0
positives = 0
negatives = 0

for i, pred in enumerate(predictions):
    # print(f"Prediction {i + 1}:")
    print(f"Predicted: {pred.mean()}")
    if pred.mean() > 0.5:
        p = 1
        positives += 1
    else:
        p = 0
        negatives += 1

    if p == y_test[i]:
        correct += 1

print(f"Accuracy: {correct / len(y_test)}")
print(f"Positives: {positives}")
print(f"Negatives: {negatives}")
