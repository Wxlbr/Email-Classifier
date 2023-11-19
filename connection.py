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

    def __init__(self, token_path='token.pickle', app_credentials_path='credentials.json'):
        '''
        Initialise the connection object
        '''

        self.__service = None
        self.__credentials = None
        self.__token_path = token_path
        self.__app_credentials_path = app_credentials_path

    def check_connected(self, reconnect=False):
        '''
        Check if the connection is valid, and reconnect if necessary
        '''

        # Check for valid credentials and service or reconnect flag
        if (not self.__credentials or not self.__credentials.valid) or (not self.__service) or (reconnect):

            # Refresh credentials
            self.update_credentials()

            # Rebuild service
            self.__service = build('gmail', self.API_VERSION, credentials=self.__credentials)

    def update_credentials(self):
        '''
        Update the credentials for the connection with the Gmail API
        '''

        # Credentials for the API
        self.__credentials = None

        # Check if token_path exists
        # The file at token_path stores the user's access and refresh tokens, and is
        # created automatically when the authorisation flow completes for the first time
        if os.path.exists(self.__token_path):
            with open(self.__token_path, 'rb') as token:
                self.__credentials = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in
        if not self.__credentials or not self.__credentials.valid:

            # Use 0Auth2.0 flow to generate credentials
            # TODO: Error handling for invalid credentials
            flow = InstalledAppFlow.from_client_secrets_file(
                self.__app_credentials_path, self.API_SCOPES)
            self.__credentials = flow.run_local_server(port=self.OAUTH_PORT)

            # Save the credentials for next time
            with open(self.__token_path, 'wb') as token:
                pickle.dump(self.__credentials, token)

    def get_user_emails(self):
        '''
        Retrieve a list of the user's emails
        '''

        self.check_connected()

        # Get a list of messages
        result = self.__service.users().messages().list(userId='me').execute()

        # Get the messages from the result
        messages = result.get('messages', [])

        # Return the messages
        return messages

    def get_email_content(self, message_id):
        '''
        Retrieve the content of an email based on its ID
        '''

        self.check_connected()

        message = self.__service.users().messages().get(userId='me', id=message_id).execute()
        payload = message['payload']

        # Get the email subject
        subject = [header['value'] for header in payload['headers'] if header['name'] == 'Subject'][0]

        # Get email body data as base64url encoded string
        body_data = payload['parts'][0]['body']['data']
        body_text = base64.urlsafe_b64decode(body_data).decode('utf-8')

        # Parse HTML content
        content = ' '.join(html.unescape(word) for word in re.findall(r'<.*?>|\b\w+\b|[.,;!?]', body_text))

        # Clean the plaintext content
        content = self._clean_content(content)

        # For debugging
        print(f"Subject: {subject}")
        print(f"Plaintext Content:\n{content}")

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

        self.check_connected()

        # Get the message
        result = self.__service.users().messages().get(userId='me', id=message_id).execute()

        # Get the labels
        labels = result.get('labelIds', [])

        # Return the labels
        return labels

    def assign_email_labels(self, message_id, labels):
        '''
        Assign labels to an email based on its ID
        '''

        self.check_connected()

        # Get the existing labels
        existing_labels = self.__service.users().labels().list(userId='me').execute()

        # Get the label IDs
        label_ids = [label['id'] for label in existing_labels['labels'] if label['name'] in labels]

        # Assign the labels to a request body
        body = {'removeLabelIds': [], 'addLabelIds': label_ids}

        # Modify the message
        self.__service.users().messages().modify(userId='me', id=message_id, body=body).execute()

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

        # For debugging
        with open('./inc/words.txt', 'w', encoding='utf-8') as f:
            json.dump(words, f)

        for word, frequency in words.items():
            if frequency > 0:
                print(f"{word}: {frequency}")

        # Return the words
        return words

    def _hot_encode(self, dictionary):
        return {key: 1 if value > 0 else value for key, value in dictionary.items()}

# Main execution
if __name__ == "__main__":
    conn = Connection()

    # Get the user's emails
    messages = conn.get_user_emails()

    # Get the content of the first email
    content = conn.get_email_content(messages[1]['id'])

    # Count word frequencies
    conn.word_counter(content)