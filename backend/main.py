import os
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow

app = FastAPI()

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"status": "ok"}

import secrets
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Database setup (SQLite for local storage)
DATABASE_URL = "sqlite:///./email_assistant.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class GmailAccount(Base):
    __tablename__ = "gmail_accounts"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    token_json = Column(String)  # Encrypted in production

Base.metadata.create_all(bind=engine)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",  # Added to allow email modification/deletion
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

REDIRECT_PATH = "/auth/google/callback"

@app.get("/auth/google")
def auth_google():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [f"http://localhost:8000{REDIRECT_PATH}"]
            }
        },
        scopes=SCOPES,
        redirect_uri=f"http://localhost:8000{REDIRECT_PATH}"
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    # Store state in session/cookie for security (omitted for brevity)
    return RedirectResponse(auth_url)

@app.get(REDIRECT_PATH)
def auth_google_callback(request: Request):
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code in callback")
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [f"http://localhost:8000{REDIRECT_PATH}"]
            }
        },
        scopes=SCOPES,
        redirect_uri=f"http://localhost:8000{REDIRECT_PATH}"
    )
    flow.fetch_token(code=code)
    credentials = flow.credentials
    from googleapiclient.discovery import build
    service = build('oauth2', 'v2', credentials=credentials)
    user_info = service.userinfo().get().execute()
    email = user_info.get("email")
    # Store or update credentials in DB
    db = SessionLocal()
    account = db.query(GmailAccount).filter(GmailAccount.email == email).first()
    if not account:
        account = GmailAccount(email=email, token_json=credentials.to_json())
        db.add(account)
    else:
        account.token_json = credentials.to_json()
    db.commit()
    db.close()
    # Redirect to frontend with success
    return RedirectResponse(f"{FRONTEND_URL}/accounts?connected={email}", status_code=status.HTTP_302_FOUND)

@app.get("/accounts")
def list_accounts():
    db = SessionLocal()
    accounts = db.query(GmailAccount).all()
    db.close()
    return [{"email": acc.email} for acc in accounts]


import datetime
import json
from fastapi import Body
from googleapiclient.discovery import build
import openai

def get_gmail_service(token_json):
    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
    return build('gmail', 'v1', credentials=creds)

def get_today_rfc3339():
    # Use datetime with timezone info
    now = datetime.datetime.now(datetime.timezone.utc)
    # Go back 24 hours to ensure we catch all of today's emails
    start = now - datetime.timedelta(hours=24)
    return start.isoformat().replace("+00:00", "Z")

def user_has_replied(service, user_email, thread_id):
    thread = service.users().threads().get(userId='me', id=thread_id).execute()
    for msg in thread.get('messages', []):
        headers = msg.get('payload', {}).get('headers', [])
        for h in headers:
            if h['name'].lower() == 'from' and user_email.lower() in h['value'].lower():
                return True
    return False

@app.get("/emails/today")
def get_today_unreplied_emails():
    db = SessionLocal()
    accounts = db.query(GmailAccount).all()
    db.close()
    result = []
    today = get_today_rfc3339()
    for acc in accounts:
        try:
            service = get_gmail_service(acc.token_json)
            # Use a more reliable query to get recent emails - include both unread and recent emails
            query = "newer_than:1d"
            print(f"Fetching emails for {acc.email} with query: {query}")
            response = service.users().messages().list(userId='me', q=query, maxResults=20).execute()
            messages = response.get('messages', [])
            print(f"Found {len(messages)} messages for {acc.email}")
            
            for m in messages:
                try:
                    msg = service.users().messages().get(userId='me', id=m['id'], format='metadata', metadataHeaders=['Subject', 'From', 'Date']).execute()
                    thread_id = msg['threadId']
                    if not user_has_replied(service, acc.email, thread_id):
                        headers = {h['name']: h['value'] for h in msg['payload'].get('headers', [])}
                        result.append({
                            'account': acc.email,
                            'id': m['id'],
                            'threadId': thread_id,
                            'subject': headers.get('Subject', ''),
                            'from': headers.get('From', ''),
                            'date': headers.get('Date', ''),
                            'snippet': msg.get('snippet', ''),
                        })
                except Exception as msg_error:
                    print(f"Error processing message {m['id']}: {str(msg_error)}")
        except Exception as e:
            print(f"Error fetching emails for {acc.email}: {str(e)}")
            continue
    return result

@app.post("/emails/delete")
def delete_email(email_data: dict = Body(...)):
    db = SessionLocal()
    acc = db.query(GmailAccount).filter(GmailAccount.email == email_data['account']).first()
    db.close()
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")
    
    service = get_gmail_service(acc.token_json)
    try:
        # Move the message to trash
        service.users().messages().trash(userId='me', id=email_data['id']).execute()
        return {"status": "success", "message": "Email moved to trash"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete email: {str(e)}")

@app.post("/emails/draft_reply")
def draft_reply(email_data: dict = Body(...)):
    db = SessionLocal()
    acc = db.query(GmailAccount).filter(GmailAccount.email == email_data['account']).first()
    db.close()
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")
    service = get_gmail_service(acc.token_json)
    msg = service.users().messages().get(userId='me', id=email_data['id'], format='full').execute()
    import base64
    payload = msg.get('payload', {})
    parts = payload.get('parts', [])
    body = ''
    if parts:
        for part in parts:
            if part.get('mimeType') == 'text/plain' and 'data' in part['body']:
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                break
    elif payload.get('body', {}).get('data'):
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    # Extract sender's name from the email headers
    headers = {h['name']: h['value'] for h in payload.get('headers', [])}
    from_header = headers.get('From', '')
    
    # Try to extract the sender's name from the From header
    sender_name = ''
    if '<' in from_header:
        # Format: "John Doe <john@example.com>"
        sender_name = from_header.split('<')[0].strip()
    elif '"' in from_header:
        # Format: "John Doe" <john@example.com>
        sender_name = from_header.split('"')[1].strip()
    else:
        # Just use the email address if we can't extract a name
        sender_name = from_header.split('@')[0] if '@' in from_header else ''
    
    # Clean up the sender name if it's just an email
    if '@' in sender_name:
        sender_name = ''
    
    # Get sender name from request if provided
    user_name = email_data.get('sender_name', '')
    
    # If no name provided in the request, use a default or extract from email
    if not user_name:
        # Extract username from email as a fallback
        email_parts = acc.email.split('@')[0].split('.')
        user_name = ' '.join(part.capitalize() for part in email_parts)
    
    prompt = f"You are an email assistant for {user_name}. You are drafting a reply to an email from {from_header}.\n\n" \
             f"If the email is professional in tone, begin with 'Dear {sender_name if sender_name else 'Sir/Madam'}' and end with 'Sincerely, {user_name}'.\n" \
             f"If the email is less formal, begin with 'Hi {sender_name.split()[0] if sender_name else ''}' and end with 'Best regards, {user_name}' or 'Thanks, {user_name}' depending on the context.\n" \
             f"Always sign off as {user_name} regardless of the formality level.\n\n" \
             f"Draft a polite, concise reply to the following email:\n\n{body}\n\nReply:"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        n=1,
        temperature=0.7,
    )
    draft = response.choices[0].message.content.strip()
    return {"draft": draft}
