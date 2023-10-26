import numpy as np
from keras.utils import np_utils

from dense import Dense, Convolutional
from activation import Tanh, Sigmoid
from loss import MSE, BinaryCrossEntropy
from reshape import Reshape
from network import train, accuracy

from sklearn.model_selection import train_test_split

import pandas as pd
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

max_height, max_width = 5, 5 # 8464, 100

clipped_word_vectors = []
for vec in word_vectors:
    if len(vec) > max_height:
        vec = vec[:max_height]
    elif len(vec) < max_height:
        vec += [np.zeros(100)] * (max_height - len(vec))

    for i in range(len(vec)):
        if len(vec[i]) > max_width:
            vec[i] = vec[i][:max_width]
        elif len(vec[i]) < max_width:
            vec[i] += [np.zeros(100)] * (max_width - len(vec[i]))

    clipped_word_vectors.append(vec)

# Convert to numpy array
X = np.array(clipped_word_vectors) # (5171, 100, 20)
Y = np.array(labels)

# Reshape input data to (5171, 1, 100, 20)
X = X.reshape(len(X), 1, max_height, max_width)

# Reshape output data to (5171, 1, 1)
Y = Y.reshape(len(Y), 1, 1)

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

network = [
    Convolutional((1, max_height, max_width), 3, 5),
    Sigmoid(),
    Reshape((5, max_height - 3 + 1, max_width - 3 + 1), (5 * (max_height - 3 + 1) * (max_width - 3 + 1), 1)),
    Dense(5 * (max_height - 3 + 1) * (max_width - 3 + 1), 100),
    Sigmoid(),
    Dense(100, 1),
    Sigmoid()
]

# train
train(network, X_train, Y_train, 100, 0.01, validation_data=(X_test, Y_test))

# accuracy
print(f"Accuracy: {accuracy(network, X_test, Y_test):.4f}%")