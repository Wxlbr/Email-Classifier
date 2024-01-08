import os
import csv
import json

import threading

from network import Network
from connection import Connection
from calc import shape


class Classifier:

    def __init__(self):
        self.conn = Connection()
        self.net = None
        self.trained = False
        self.accuracy = 0
        self.stop_event = threading.Event()

    def stop_classification(self):
        # print('Stopping classification...')
        self.stop_event.set()

    def start_classification_thread(self, n=-1):
        # Reset the stop event in case it was set before
        self.stop_event.clear()

        # Start classification in a separate thread
        thread = threading.Thread(target=self.classify_emails, args=(n,))
        thread.start()

    def main(self, loop=False, n=-1):
        '''
        Classify the emails
        '''

        # print('Loading network...')

        if self.net is None:
            # Load default network
            self.load_network(file_path='./inc/model.json')

        assert self.trained, 'Network is not trained.'

        while loop and not self.stop_event.is_set():
            self.classify_emails(n=n)

    def classify_emails(self, n=-1):

        # print('Classifying emails...')

        # Get the user's emails
        messages = self.conn.get_user_emails()

        # Classify each email
        for i, message in enumerate(messages):
            if i == n or self.stop_event.is_set():
                # print('Stopping classification...')
                break

            # print('Classifying email', i)

            self.classify_email(message['id'], assign_label=False)

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

    def start_training_thread(self, network_id, layers, epochs, socketio):
        # Reset the stop event in case it was set before
        if self.net:
            self.net.clear_stop_event()

        # Start training in a separate thread
        thread = threading.Thread(target=self.train_network_thread, args=(
            network_id, layers, epochs, socketio,))
        thread.start()

    def train_network_thread(self, network_id, layers, epochs, socketio):
        for layer in layers.values():
            self.add_layer(layer['layerConfig'])

        print('Training network')

        self.train_network(
            epochs=epochs, socketio=socketio, netId=network_id)

        # Get network from file
        with open(f'./inc/networks/{network_id}.json', 'r', encoding='utf-8') as f:
            network = json.load(f)

        # Save network to the dictionary (network_id will already be in dictionary)
        network['network'] = self.net.info()
        network['network']['accuracy'] = self.accuracy

        # Save network to file
        with open(f'./inc/networks/{network_id}.json', 'w', encoding='utf-8') as f:
            json.dump(network, f, indent=4)

    def stop_training(self):
        '''
        Stop the training of the network
        '''

        if self.net is not None:
            self.net.stop()

    def train_network(self, X=None, Y=None, epochs=1, socketio=None, netId=None):
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
        self.trained, self.accuracy = self.net.train(X_train, Y_train, epochs=epochs, validation_data=(
            X_test, Y_test), socketio=socketio, netId=netId)

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

    def load_network(self, network_dictionary=None, file_path=None):
        '''
        Load a trained network from a file or dictionary
        '''

        # Only one of the parameters can be provided
        assert (network_dictionary is None) != (
            file_path is None), 'Only one of the parameters can be provided.'

        if file_path is not None:
            assert os.path.exists(file_path), 'File does not exist.'

        if network_dictionary is not None:
            assert isinstance(network_dictionary,
                              dict), 'Network is not a dictionary.'

        self.net = Network()
        self.net.load(network_dictionary=network_dictionary,
                      file_path=file_path)
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
