from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import base64
from google.oauth2.credentials import Credentials
import pickle

def authenticate():
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.labels', 'https://www.googleapis.com/auth/gmail.modify']

    token_path = 'token.pickle' # User access token
    creds = None

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=8080)
        # Save the credentials for next time
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return creds

def get_user_emails(service):
    result = service.users().messages().list(userId='me').execute()
    messages = result.get('messages', [])

    return messages

def get_email_labels(service, message_id):
    labels = service.users().messages().get(userId='me', id=message_id).execute()['labelIds']

    return labels

def assign_email_labels(service, message_id, labels):
    existing_labels = service.users().labels().list(userId='me').execute()
    label_ids = [label['id'] for label in existing_labels['labels'] if label['name'] in labels]

    body = {'removeLabelIds': [], 'addLabelIds': label_ids}
    service.users().messages().modify(userId='me', id=message_id, body=body).execute()

# Main execution
if __name__ == "__main__":
    # Authenticate
    credentials = authenticate()
    service = build('gmail', 'v1', credentials=credentials)

    # Get user emails
    user_emails = get_user_emails(service)
    print("User Emails:", user_emails)

    # Choose a message_id from user_emails
    if user_emails:
        message_id = user_emails[0]['id']

        # Get email labels
        email_labels = get_email_labels(service, message_id)
        print("Email Labels:", email_labels)

        # Assign email labels
        label_ids = ['Safe']
        assign_email_labels(service, message_id, label_ids)
        print("Labels assigned successfully.")
