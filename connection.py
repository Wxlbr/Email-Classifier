import os
import re
import html
import json
import base64
import pickle

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class Connection:
    OAUTH_PORT = 8080
    API_VERSION = 'v1'
    API_SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.labels',
        'https://www.googleapis.com/auth/gmail.modify'
    ]

    def __init__(self, token_path='./inc/credentials/token.pickle', app_credentials_path='./inc/credentials/credentials.json'):
        '''
        Initialise the connection object
        '''

        self._service = None
        self._credentials = None
        self._token_path = token_path
        self._app_credentials_path = app_credentials_path

    def _check_connected(self):
        '''
        Check if the connection is valid, and reconnect if necessary
        '''

        # Check for valid credentials and service or reconnect flag
        if (not self._credentials or not self._credentials.valid) or (
                not self._service):

            # Refresh credentials
            self._update_credentials()

            # Rebuild service
            self._service = build(
                'gmail', self.API_VERSION, credentials=self._credentials)

    def _update_credentials(self):
        '''
        Update the credentials for the connection with the Gmail API
        '''

        # Credentials for the API
        self._credentials = None

        # Check if token_path exists
        # The file at token_path stores the user's access and refresh tokens, and is
        # created automatically when the authorisation flow completes for the first time
        if os.path.exists(self._token_path):
            with open(self._token_path, 'rb') as token:
                self._credentials = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in
        if not self._credentials or not self._credentials.valid:

            # Use 0Auth2.0 flow to generate credentials
            flow = InstalledAppFlow.from_client_secrets_file(
                self._app_credentials_path, self.API_SCOPES)
            self._credentials = flow.run_local_server(port=self.OAUTH_PORT)

            # Save the credentials for next time
            with open(self._token_path, 'wb') as token:
                pickle.dump(self._credentials, token)

    def get_user_emails(self):
        '''
        Retrieve a list of the user's emails
        '''

        self._check_connected()

        # Get a list of messages
        result = self._service.users().messages().list(userId='me').execute()

        # Get the messages from the result
        messages = result.get('messages', [])

        # Return the messages
        return messages

    def get_email_content(self, message_id):
        '''
        Retrieve the content of an email based on its ID
        '''

        self._check_connected()

        message = self._service.users().messages().get(
            userId='me', id=message_id).execute()
        payload = message['payload']

        # Get email body data as base64url encoded string
        content = ''
        if 'parts' in payload and payload['parts'][0]['body']['size'] > 0:
            body_data = payload['parts'][0]['body']['data']
            body_text = base64.urlsafe_b64decode(body_data).decode('utf-8')

            # Parse HTML content
            content = ' '.join(html.unescape(word) for word in re.findall(
                r'<.*?>|\b\w+\b|[.,;!?]', body_text))

        # Clean the plaintext content
        content = self._clean_content(content)

        # Return the plaintext content
        return content

    def _clean_content(self, content):
        '''
        Clean and return the content
        '''

        cleaned_content = []
        for word in content.lower().split():
            if not word.isalnum():
                word = ''.join(char for char in word if char.isalnum())
            if word:
                cleaned_content.append(word)

        content = ' '.join(cleaned_content)

        return content

    def get_email_labels(self, message_id):
        '''
        Retrieve the labels of an email based on its ID
        '''

        self._check_connected()

        # Get the message
        result = self._service.users().messages().get(
            userId='me', id=message_id).execute()

        # Get the labels
        labels = result.get('labelIds', [])

        # Return the labels
        return labels

    def _create_label(self, label_name):
        '''
        Create a label with the given name
        '''

        print(f'Creating label: {label_name}')

        self._check_connected()

        # Create the label
        label = {'name': label_name, 'messageListVisibility': 'show',
                 'labelListVisibility': 'labelShow'}

        # Create the label
        self._service.users().labels().create(userId='me', body=label).execute()

    def _get_label_id(self, label_name):
        '''
        Get the ID of a label with the given name
        '''

        self._check_connected()

        # Get the existing labels
        existing_labels = self._service.users().labels().list(userId='me').execute()

        # Get the label ID
        label_id = next(
            (label['id'] for label in existing_labels['labels'] if label['name'] == label_name), None)

        # If the label doesn't exist, create it
        if not label_id:
            self._create_label(label_name)
            label_id = self._get_label_id(label_name)

        # Return the label ID
        return label_id

    def _get_label_name(self, label_id):
        '''
        Get the name of a label with the given ID
        '''

        self._check_connected()

        # Get the existing labels
        existing_labels = self._service.users().labels().list(userId='me').execute()

        # Get the label name
        label_name = next(
            (label['name'] for label in existing_labels['labels'] if label['id'] == label_id), None)

        # Return the label name
        return label_name

    def remove_email_labels(self, message_id, labels):
        '''
        Remove labels from an email based on its name
        '''

        self._check_connected()

        # Get the label IDs
        label_ids = [self._get_label_id(label) for label in labels]

        # Remove the labels from a request body
        body = {
            'removeLabelIds': label_ids,
            'addLabelIds': []
        }

        # Modify the message
        self._service.users().messages().modify(
            userId='me', id=message_id, body=body).execute()

    def assign_email_labels(self, message_id, labels):
        '''
        Assign labels to an email based on its name
        '''

        self._check_connected()

        # Check that both safe and unsafe are not in the labels
        assert not ('Safe' in labels and 'Unsafe' in labels)

        # Check that the corresponding labels have not already been assigned
        existing_labels = self.get_email_labels(message_id)

        existing_labels = [self._get_label_name(
            label_id) for label_id in existing_labels]

        # If the labels are already assigned, remove them
        if 'Safe' in labels and 'Unsafe' in existing_labels:
            self.remove_email_labels(message_id, ['Unsafe'])
        elif 'Unsafe' in labels and 'Safe' in existing_labels:
            self.remove_email_labels(message_id, ['Safe'])

        print('Getting label IDs')

        # Get the label IDs
        label_ids = [self._get_label_id(label) for label in labels]

        # Assign the labels to a request body
        body = {'removeLabelIds': [], 'addLabelIds': label_ids}

        # Modify the message
        self._service.users().messages().modify(
            userId='me', id=message_id, body=body).execute()

    def word_counter(self, plaintext):
        '''
        Return a dictionary of words and their hot encoded frequencies from the plaintext
        '''

        # Get the words
        with open('inc/words.json', 'r', encoding='utf-8') as f:
            words = {word: 0 for word in json.load(f)}

        # Count the frequencies
        for word in plaintext.split():
            if word in words:
                words[word] += 1

        # Hot encode the frequencies
        words = self._hot_encode(words)

        # Return the words
        return words

    def _hot_encode(self, dictionary):
        return {key: 1 if value > 0 else value for key, value in dictionary.items()}

    def email_has_label(self, message_id):
        '''
        Check if an email has a label
        '''

        self._check_connected()

        # Get the labels
        labels = self.get_email_labels(message_id)

        check_labels = [self._get_label_id(label)
                        for label in ['Safe', 'Unsafe']]

        # Check if the email has a label
        return any(label in labels for label in check_labels)

    def unclassify_emails(self):
        '''
        Remove all labels from all emails 
        '''

        self._check_connected()

        # Get the user's emails
        messages = self.get_user_emails()

        # Remove the labels from each email
        for _, message in enumerate(messages):
            self.remove_email_labels(message['id'], ['Safe', 'Unsafe'])


if __name__ == "__main__":
    conn = Connection()

    # Get the user's emails
    messages = conn.get_user_emails()

    # Get the content of the first email
    for i in range(20):
        content = conn.get_email_content(messages[i]['id'])

    # Count word frequencies
    # conn.word_counter(content)
