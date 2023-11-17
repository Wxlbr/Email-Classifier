import os
import re
import html
import base64
import pickle

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

class Connection:

    def __init__(self):
        self.__SCOPES = [
                            'https://www.googleapis.com/auth/gmail.readonly',
                            'https://www.googleapis.com/auth/gmail.labels',
                            'https://www.googleapis.com/auth/gmail.modify'
                        ]

        self.__service = None
        self.__credentials = None

    def check_connected(self, reconnect=False):

        # Check for valid credentials and service or reconnect flag
        if (not self.__credentials or not self.__credentials.valid) or (not self.__service) or (reconnect):

            # Refresh credentials
            self.update_credentials()

            # Rebuild service
            self.__service = build('gmail', 'v1', credentials=self.__credentials)

    def update_credentials(self):

        # The file at token_path stores the user's access and refresh tokens, and is
        # created automatically when the authorisation flow completes for the first time
        token_path = 'token.pickle'

        # Credentials for the API
        self.__credentials = None

        # Check if token_path exists
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                self.__credentials = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in
        if not self.__credentials or not self.__credentials.valid:

            # Use 0Auth2.0 flow to generate credentials
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', self.__SCOPES)
            self.__credentials = flow.run_local_server(port=8080)

            # Save the credentials for next time
            with open(token_path, 'wb') as token:
                pickle.dump(self.__credentials, token)

    def get_user_emails(self):
        self.check_connected()

        # Get a list of messages
        result = self.__service.users().messages().list(userId='me').execute()

        # Get the messages from the result
        messages = result.get('messages', [])

        # Return the messages
        return messages

    def get_email_content(self, message_id):
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
        cleaned_content = []
        for word in content.lower().split():
            if not word.isalnum():
                word = ''.join(char for char in word if char.isalnum())
            if word:
                cleaned_content.append(word)

        content = ' '.join(cleaned_content)

        print("Subject:", subject)
        print(f"Plaintext Content:\n{content}")

    def get_email_labels(self, message_id):
        self.check_connected()

        # Get the message
        result = self.__service.users().messages().get(userId='me', id=message_id).execute()

        # Get the labels
        labels = result.get('labelIds', [])

        # Return the labels
        return labels

    def assign_email_labels(self, message_id, labels):
        self.check_connected()

        # Get the existing labels
        existing_labels = self.__service.users().labels().list(userId='me').execute()

        # Get the label IDs
        label_ids = [label['id'] for label in existing_labels['labels'] if label['name'] in labels]

        # Assign the labels to a request body
        body = {'removeLabelIds': [], 'addLabelIds': label_ids}

        # Modify the message
        self.__service.users().messages().modify(userId='me', id=message_id, body=body).execute()

# Main execution
if __name__ == "__main__":
    conn = Connection()

    # Get the user's emails
    messages = conn.get_user_emails()

    # Get the content of the first email
    conn.get_email_content(messages[0]['id'])
