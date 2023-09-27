import pandas as pd
import numpy as np
import tensorflow as tf

# Read data from inc/kaggleDataset.csv
with open('inc/kaggleDataset.csv', 'r') as f:
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

# Define a function for text preprocessing using a hash function
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

total_text = np.concatenate((train_text, test_text), axis=0)

print(train_text.shape, test_text.shape)

# Pad sequences to make them of equal length
max_sequence_length = max(len(seq) for seq in total_text)
train_text = tf.keras.preprocessing.sequence.pad_sequences(train_text) #, maxlen=max_sequence_length)
test_text = tf.keras.preprocessing.sequence.pad_sequences(test_text) # , maxlen=max_sequence_length)

print(train_text.shape, test_text.shape)

# Create an LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=max_sequence_length),
    tf.keras.layers.LSTM(units=100),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_text, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_text, test_labels)

# Print the accuracy
print("Test Accuracy:", accuracy)