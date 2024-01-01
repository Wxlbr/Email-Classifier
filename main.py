import os
import csv

from network import Network
from connection import Connection
from calc import shape


class Classifier:

    def __init__(self):
        self.conn = Connection()
        self.net = None
        self.trained = False

    def main(self, loop=False, n=-1, stop_event=None):
        '''
        Classify the emails
        '''

        if self.net is None:
            # Load default network
            self.load_network(file_path='./inc/model.json')

        assert self.trained, 'Network is not trained.'

        # FOR DEBUGGING
        # self.conn.unclassify_emails()

        # self.classify_emails(n=n, stop_event=stop_event)

        while loop and (stop_event is None or not stop_event.is_set()):
            self.classify_emails(n=n, stop_event=stop_event)

    def classify_emails(self, n=-1, stop_event=None):

        print('Classifying emails...')

        # Get the user's emails
        messages = self.conn.get_user_emails()

        # Classify each email
        for i, message in enumerate(messages):
            if i == n and n != -1 or stop_event is not None and stop_event.is_set():
                break

            self.classify_email(message['id'], assign_label=True)

    def test_network(self):
        '''
        Function for testing the network (debugging)
        '''

        if self.net is None:
            self.load_network(file_path='./inc/model.json')

        assert self.trained, 'Network is not trained.'

        # Get the user's emails
        messages = self.conn.get_user_emails()

        # Classify each email
        for _, message in enumerate(messages):

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

    def classify_email(self, message_id, assign_label=True):
        '''
        Get the content of the email, classify it and assign the label
        '''

        assert self.net is not None, 'No network found.'
        assert self.trained, 'Network is not trained.'

        # Check if the email has already been classified
        if self.conn.email_has_label(message_id):
            print('Skip')
            return

        # Get the content of the first email
        content = self.conn.get_email_content(message_id)

        # Count word frequencies
        content = self.conn.word_counter(content)

        # Convert to list of values
        content = [[content[key]] for key in content]

        # Predict the class label of the email
        predicted_class = round(self.net.predict(content)[0][0])

        # Assign the label
        label = 'Safe' if predicted_class == 0 else 'Unsafe'

        # Assign the label to the email
        if assign_label:
            self.conn.assign_email_labels(message_id, [label])

    def train_network(self, X=None, Y=None, socketio=None, netId=None):
        '''
        Train the network
        '''

        print('Training network...')

        # Use default training data if none is provided
        if X is None or Y is None:
            X, Y = self._default_training_data()

        # Create a default network if none exists
        if self.net is None:
            print('No network found. Creating a default network.')
            self.net = self._default_network(len(X[0]), 1)

        assert self._check_data_compatability(
            X, Y), 'Data is not compatible with the current network.'

        # Split into train and test sets
        X_train, X_test, Y_train, Y_test = self._train_test_split(
            X, Y, test_size=0.2)

        # Train the network
        self.net.train(X_train, Y_train, validation_data=(
            X_test, Y_test), socketio=socketio, netId=netId)

        self.trained = True

    def _train_test_split(self, X, Y, test_size=0.2):
        '''
        Split the data into train and test sets
        '''

        assert len(X) == len(Y) and len(X) > 0

        train_size = int(len(X) * (1 - test_size))
        test_size = len(X) - train_size

        assert train_size > 0 and test_size > 0

        X_train, Y_train = X[:train_size], Y[:train_size]
        X_test, Y_test = X[train_size:], Y[train_size:]

        return X_train, X_test, Y_train, Y_test

    def load_network(self, file_path):
        '''
        Load the network from file
        '''

        assert os.path.exists(file_path), 'File does not exist.'

        self.net = Network()
        self.net.load(file_path)
        self.trained = True

    def add_layer(self, layer):
        '''
        Add a layer to the network
        '''
        if self.net is None:
            self.net = Network()

        self.net.add_layer(layer)

    def _check_data_compatability(self, X, Y):
        '''
        Check if the data is compatible with the current network
        '''

        # Data is not compatible if there is no network
        if self.net is None:
            return False

        # Data is not compatible if the input and output sizes do not match that of the network
        return self.net.layers[0].input_size == len(X[0]) and self.net.layers[-1].output_size == 1

    def _default_network(self, input_size, output_size):
        '''
        Return the default network
        '''

        net = Network()

        net.add_layer({"inputSize": str(input_size), "outputSize": str(output_size),
                      "type": "recurrent", "activation": "sigmoid"})

        return net

    def _default_training_data(self):
        '''
        Return the default training data
        '''

        with open('./inc/emailsHotEncoding.csv', 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            next(r)  # Skip header
            data = list(r)

        # Extract features and labels
        X = [list([float(value)] for value in row[:-1]) for row in data]
        Y = [int(row[-1]) for row in data]

        print(shape(X), shape(Y))

        return X, Y


if __name__ == "__main__":

    classifier = Classifier()

    classifier.main()

    # classifier.train_network()
    # classifier.test_network()
