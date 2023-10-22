from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Wrapper class for Word2Vec
class Word2VecWrapper:
    def __init__(self, sentences, workers=30):
        self.model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=workers)

    def vectorise(self, text, pad=None):
        text = self.normalise(text)
        words = text.split(' ')
        vectors = []
        for word in words:
            num = self.convert(word)
            if num != 0.0:
                vectors.append(num)
        if pad is not None:
            if len(vectors) < pad:
                vectors.extend([0.0] * (pad - len(vectors)))
            elif len(vectors) > pad:
                vectors = vectors[:pad]
        return np.array(vectors)

    def normalise(self, string):
        string = string.lower()
        string.replace('\n', ' ')
        for char in string:
            if not char.isalnum() and char != ' ':
                string = string.replace(char, '')
        return string

    def convert(self, word):
        if word in self.model.wv:
            arr = np.array(self.model.wv[word])
        else:
            arr = np.zeros(100, dtype=np.float32)
        return np.mean(arr, axis=0) # Average the word vectors to get a single value for the word

# Load the data
data = pd.read_csv('inc/kaggleDataset.csv')
text = data['text'].tolist()

# Get the labels
labels = data.pop('label')
labels = labels.map({'ham': 0, 'spam': 1})

model = Word2VecWrapper([sentence.split() for sentence in text])

# Convert the text to word vectors
padded_vectors = np.array([model.vectorise(sentence, pad=100) for sentence in text])

# Reduce the data size by sampling to 1000 vectors of length 100 each of single words
sample_size = 1000
sample_indices = np.random.choice(len(padded_vectors), sample_size, replace=False)

padded_vectors_sampled = padded_vectors[sample_indices]
labels_sampled = labels[sample_indices]

print(padded_vectors_sampled)

# Split the sampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_vectors_sampled, labels_sampled, test_size=0.2, random_state=42)