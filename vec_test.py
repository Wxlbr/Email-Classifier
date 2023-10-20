import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

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

# Word2Vec
model = Word2Vec(sentences=text, vector_size=100, window=5, min_count=1, workers=4)
words = list(model.wv.index_to_key)

# Convert text to word vectors
word_vectors = []
for sentence in text:
    sentence_vectors = [model.wv[word] for word in sentence if word in model.wv]
    if len(sentence_vectors) > 0:
        word_vectors.append(sentence_vectors)
    else:
        word_vectors.append([np.zeros(100)])

# Padding the sequences to make them of equal length
padded_vectors = tf.keras.preprocessing.sequence.pad_sequences(word_vectors, padding='post')

# Reduce the data size by sampling
sample_size = 3000  # Adjust this as needed
# sample_size = len(padded_vectors)
# print(sample_size)
sample_indices = np.random.choice(len(padded_vectors), sample_size, replace=False)

padded_vectors_sampled = padded_vectors[sample_indices]
labels_sampled = labels.iloc[sample_indices]

# list size
list_size = 100
new_padded_vectors = []
for i in range(sample_size):
    new_padded_vectors.append(padded_vectors_sampled[i][:list_size])

padded_vectors_sampled = np.array(new_padded_vectors)

# Split the sampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_vectors_sampled, labels_sampled, test_size=0.2, random_state=42)

# Build and train an RNN model
# model = tf.keras.Sequential([
#     tf.keras.layers.Masking(mask_value=np.zeros(100), input_shape=(list_size, 100)),
#     tf.keras.layers.LSTM(64),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=np.zeros(100), input_shape=(list_size, 100)),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))