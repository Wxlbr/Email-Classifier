import os

import pandas as pd

from network import Network
from split import train_test_split
from connection import Connection


class Classifier:

    def __init__(self):
        self.conn = Connection()
        self.net = None

    def main(self, n=1):
        '''
        Classify the emails
        '''

        if self.net is None:
            self.load_network(file_path='./inc/model.json')

        # Get the user's emails
        messages = self.conn.get_user_emails()

        # Classify each email
        for i, message in enumerate(messages):
            if i == n:
                break

            self.classify_email(message['id'])

    def test_network(self):
        '''
        Test the network
        '''

        if self.net is None:
            self.load_network(file_path='./inc/model.json')

        # Get the user's emails
        messages = self.conn.get_user_emails()

        # Classify each email
        for i, message in enumerate(messages):

            # Get the content of the first email
            content = self.conn.get_email_content(message['id'])

            # Count word frequencies
            content = self.conn.word_counter(content)

            # Convert to list of values
            content = [content[key] for key in content]

            # Predict the class label of the email
            predicted_class = round(self.net.predict(content)[0][0])

            print(predicted_class, end='')

        print()

    def classify_email(self, message_id):
        '''
        Get the content of the email, classify it and assign the label
        '''

        # Get the content of the first email
        content = self.conn.get_email_content(message_id)

        # Count word frequencies
        content = self.conn.word_counter(content)

        # Convert to list of values
        content = [content[key] for key in content]

        # Predict the class label of the email
        predicted_class = round(self.net.predict(content)[0][0])

        # Assign the label
        label = 'Safe' if predicted_class == 0 else 'Unsafe'

        # Assign the label to the email
        self.conn.assign_email_labels(message_id, [label])

    # TODO: Add check for network layer structure
    def train_network(self, X=None, Y=None):
        '''
        Train the network
        '''

        if X is None or Y is None:
            X, Y = self._default_training_data()

        # max_height = X.shape[1]
        # max_width = 1

        # Split into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        # Train the network
        self.net.train(X_train, Y_train, validation_data=(X_test, Y_test))

    def load_network(self, file_path):
        '''
        Load the network from file
        '''

        if os.path.exists(file_path):
            self.net = Network()
            self.net.load(file_path)

        else:
            raise Exception('File does not exist.')

    def _default_training_data(self):
        '''
        Return the default training data
        '''

        # Read data from inc/kaggleDataset.csv
        with open('./inc/emailsHotEncoding.csv', 'r', encoding='utf-8') as f:
            emails = pd.read_csv(f)

        X = emails.drop('Prediction', axis=1).values
        Y = emails['Prediction'].values

        X = X.reshape((X.shape[0], X.shape[1], 1))
        Y = Y.reshape((Y.shape[0], 1, 1))

        return X, Y


if __name__ == "__main__":

    classifier = Classifier()

    classifier.main()
    # classifier.test_network()
