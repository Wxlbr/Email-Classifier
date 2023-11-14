from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import base64
from google.oauth2.credentials import Credentials
import pickle
from bs4 import BeautifulSoup

def authenticate():

    # If modifying these scopes, delete the file token.pickle
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.labels', 'https://www.googleapis.com/auth/gmail.modify']

    # The file at token_path stores the user's access and refresh tokens, and is
    # created automatically when the authorisation flow completes for the first time
    token_path = 'token.pickle'

    # Credentials for the API
    creds = None

    # Check if token_path exists
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:

        # Use 0Auth2.0 flow to generate credentials
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=8080)

        # Save the credentials for next time
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    # Return the authenticated Gmail API credentials
    return creds

def get_user_emails(service):

    # Get a list of messages
    result = service.users().messages().list(userId='me').execute()

    # Get the messages from the result
    messages = result.get('messages', [])

    # Return the messages
    return messages

def get_email_content(service, message_id):
    message = service.users().messages().get(userId='me', id=message_id).execute()
    payload = message['payload']
    headers = payload['headers']
    subject = [header['value'] for header in headers if header['name'] == 'Subject'][0]

    # The email body is in the 'data' field of the 'body' property of the 'payload'
    # TODO: KeyError: 'data'
    body_data = payload['body']['data']
    body_text = base64.urlsafe_b64decode(body_data).decode('utf-8')

    # Parse HTML content using BeautifulSoup to extract plaintext
    soup = BeautifulSoup(body_text, 'html.parser')
    plaintext_content = soup.get_text(separator='\n')

    # Remove empty lines
    plaintext_content = '\n'.join([line for line in plaintext_content.split('\n') if line.strip()])

    # Remove no alphanumeric characters
    plaintext_content = ' '.join(x if x.isalnum() else "" for x in plaintext_content.split())

    # Remove duplicate spaces
    plaintext_content = ' '.join(plaintext_content.split())

    # Remove leading and trailing spaces
    plaintext_content = plaintext_content.strip()

    # Lowercase
    plaintext_content = plaintext_content.lower()

    print("Subject:", subject)
    print(f"Plaintext Content:\n{plaintext_content}")

def get_email_labels(service, message_id):

    # Get the message
    result = service.users().messages().get(userId='me', id=message_id).execute()

    # Get the labels
    labels = result.get('labelIds', [])

    # Return the labels
    return labels

def assign_email_labels(service, message_id, labels):

    # Get the existing labels
    existing_labels = service.users().labels().list(userId='me').execute()

    # Get the label IDs
    label_ids = [label['id'] for label in existing_labels['labels'] if label['name'] in labels]

    # Assign the labels to a request body
    body = {'removeLabelIds': [], 'addLabelIds': label_ids}

    # Modify the message
    service.users().messages().modify(userId='me', id=message_id, body=body).execute()

# Main execution
if __name__ == "__main__":
    # Authenticate
    credentials = authenticate()
    service = build('gmail', 'v1', credentials=credentials)

    # Get user emails
    user_emails = get_user_emails(service)
    # print("User Emails:", user_emails)

    # Choose a message_id from user_emails
    if user_emails:
        message_id = user_emails[0]['id']

        # Get email labels
        # email_labels = get_email_labels(service, message_id)
        # print("Email Labels:", email_labels)

        # Assign email labels
        # label_ids = ['Safe']
        # assign_email_labels(service, message_id, label_ids)
        # print("Labels assigned successfully.")

        # Get email content
        get_email_content(service, message_id)
