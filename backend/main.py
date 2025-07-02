import os
import threading
from typing import Dict
from fastapi import FastAPI, Depends, HTTPException, Request, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow
from email_classifier import EmailClassifier

app = FastAPI()

# Initialize the classifier globally
email_classifier = EmailClassifier()

# Flag to track if we've attempted to initialize the model
_model_initialized = False

# CORS for frontend (local and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", os.getenv("FRONTEND_URL", "")],
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

# Ensure classifiers are initialized for all accounts
def initialize_classifiers_if_needed():
    global _model_initialized
    
    if _model_initialized:
        return
        
    try:
        db = SessionLocal()
        accounts = db.query(GmailAccount).all()
        db.close()
        
        if accounts:
            for account in accounts:
                # Get model path for this account
                model_path = email_classifier._get_model_path(account.email)
                
                # Check if model exists for this account
                if os.path.exists(model_path):
                    print(f"Using existing classifier model for {account.email}")
                    # Model will be loaded when needed
                else:
                    # Train a model for this account
                    print(f"Auto-training classifier for {account.email}")
                    email_classifier.train_model(account.email, account.token_json)
                
        _model_initialized = True
    except Exception as e:
        print(f"Error initializing classifiers: {str(e)}")

@app.get("/emails/today")
def get_today_unreplied_emails():
    db = SessionLocal()
    accounts = db.query(GmailAccount).all()
    db.close()
    result = []
    today = get_today_rfc3339()
    
    # Initialize classifiers for all accounts if needed
    initialize_classifiers_if_needed()
    
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
                        email_data = {
                            'account': acc.email,
                            'id': m['id'],
                            'threadId': thread_id,
                            'subject': headers.get('Subject', ''),
                            'from': headers.get('From', ''),
                            'date': headers.get('Date', ''),
                            'snippet': msg.get('snippet', ''),
                        }
                        
                        # Classify the email using our account-specific model
                        try:
                            # Add the account email to the data for account-specific classification
                            email_data['account'] = acc.email
                            category = email_classifier.classify_email(email_data)
                            email_data['category'] = category
                            
                            # If classified as trash, automatically move to trash in Gmail
                            if category == 'trash':
                                try:
                                    print(f"Auto-moving email {m['id']} to trash for {acc.email}")
                                    service.users().messages().trash(userId='me', id=m['id']).execute()
                                    email_data['auto_trashed'] = True
                                except Exception as trash_error:
                                    print(f"Error auto-trashing email {m['id']}: {str(trash_error)}")
                                    email_data['auto_trashed'] = False
                            else:
                                email_data['auto_trashed'] = False
                        except Exception as classify_error:
                            print(f"Error classifying email {m['id']} for {acc.email}: {str(classify_error)}")
                            email_data['category'] = 'regular'  # Default category
                            email_data['auto_trashed'] = False
                            
                        result.append(email_data)
                except Exception as msg_error:
                    print(f"Error processing message {m['id']}: {str(msg_error)}")
        except Exception as e:
            print(f"Error fetching emails for {acc.email}: {str(e)}")
            continue
    return result

@app.post("/emails/delete")
def delete_email(email_data: dict = Body(...)):
    """Move an email to trash in Gmail"""
    try:
        email_id = email_data.get('email_id')
        account_email = email_data.get('account_email')
        
        if not email_id or not account_email:
            raise HTTPException(status_code=400, detail="Missing email_id or account_email")
        
        db = SessionLocal()
        acc = db.query(GmailAccount).filter(GmailAccount.email == account_email).first()
        db.close()
        
        if not acc:
            raise HTTPException(status_code=404, detail="Account not found")
        
        service = get_gmail_service(acc.token_json)
        service.users().messages().trash(userId='me', id=email_id).execute()
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete email: {str(e)}")
        
@app.post("/emails/recover")
def recover_email(email_data: dict = Body(...)):
    """Recover an email from trash in Gmail"""
    try:
        email_id = email_data.get('email_id')
        account_email = email_data.get('account_email')
        
        if not email_id or not account_email:
            raise HTTPException(status_code=400, detail="Missing email_id or account_email")
        
        db = SessionLocal()
        acc = db.query(GmailAccount).filter(GmailAccount.email == account_email).first()
        db.close()
        
        if not acc:
            raise HTTPException(status_code=404, detail="Account not found")
        
        service = get_gmail_service(acc.token_json)
        # Untrash the message and move it back to inbox
        service.users().messages().untrash(userId='me', id=email_id).execute()
        
        # Update the category to 'regular' in our classifier too
        # This helps the classifier learn from this correction
        retrain_thread = threading.Thread(
            target=email_classifier.train_model,
            args=(acc.email, acc.token_json)
        )
        retrain_thread.daemon = True
        retrain_thread.start()
        
        return {"status": "success", "message": "Email recovered from trash"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recover email: {str(e)}")

@app.post("/train/classifier")
def train_classifier(request: Dict):
    """Train the email classifier using the specified email account"""
    account_email = request.get('account_email')
    if not account_email:
        raise HTTPException(status_code=400, detail="Missing account_email")
    
    db = SessionLocal()
    account = db.query(GmailAccount).filter(GmailAccount.email == account_email).first()
    db.close()
    
    if not account:
        raise HTTPException(status_code=404, detail=f"Account {account_email} not found")
    
    try:
        # Pass account email to train model for this specific account
        email_classifier.train_model(account.email, account.token_json)
        return {"status": "success", "message": "Classifier trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training classifier: {str(e)}")

@app.post("/emails/classify")
def classify_email(email_data: dict):
    """Classify an email using the account-specific trained model"""
    try:
        # Ensure account email is included for account-specific classification
        if 'account' not in email_data:
            raise HTTPException(status_code=400, detail="Missing account email in request")
            
        category = email_classifier.classify_email(email_data)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to classify email: {str(e)}")
        
@app.get("/emails/classify/{email_id}")
def classify_specific_email(email_id: str, account_email: str):
    """Classify a specific email by ID"""
    db = SessionLocal()
    account = db.query(GmailAccount).filter(GmailAccount.email == account_email).first()
    db.close()
    
    if not account:
        raise HTTPException(status_code=404, detail=f"Account {account_email} not found")
    
    try:
        service = get_gmail_service(account.token_json)
        msg = service.users().messages().get(userId='me', id=email_id, format='metadata',
                                            metadataHeaders=['Subject', 'From']).execute()
        
        headers = {h['name']: h['value'] for h in msg['payload']['headers']}
        email_data = {
            'id': email_id,
            'subject': headers.get('Subject', ''),
            'from': headers.get('From', ''),
            'snippet': msg.get('snippet', ''),
            'account': account_email  # Add account email for account-specific classification
        }
        
        category = email_classifier.classify_email(email_data)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying email: {str(e)}")

@app.post("/emails/update_category")
def update_email_category(request: Dict):
    """Update email category and trigger retraining"""
    email_id = request.get('email_id')
    account_email = request.get('account_email')
    category = request.get('category')
    
    if not email_id or not account_email or not category:
        raise HTTPException(status_code=400, detail="Missing required fields")
        
    if category not in ['important', 'regular', 'trash']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    db = SessionLocal()
    account = db.query(GmailAccount).filter(GmailAccount.email == account_email).first()
    db.close()
    
    if not account:
        raise HTTPException(status_code=404, detail=f"Account {account_email} not found")
    
    try:
        service = get_gmail_service(account.token_json)
        
        # Apply gmail labels based on category
        if category == 'important':
            # Star the email
            service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'addLabelIds': ['STARRED']}
            ).execute()
            # Remove from trash if it was there
            service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['TRASH']}
            ).execute()
        elif category == 'trash':
            # Move to trash
            service.users().messages().trash(userId='me', id=email_id).execute()
            # Remove star if it was starred
            service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['STARRED']}
            ).execute()
        else:  # regular
            # Remove star and remove from trash
            service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['STARRED', 'TRASH']}
            ).execute()
        
        # Retrain the classifier for this account in a background thread
        retrain_thread = threading.Thread(
            target=email_classifier.train_model,
            args=(account_email, account.token_json)
        )
        retrain_thread.daemon = True
        retrain_thread.start()
        
        return {"status": "success", "message": f"Email {email_id} category updated to {category}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating email category: {str(e)}")

@app.post("/emails/draft_reply")
def draft_reply(email_data: dict = Body(...)):
    """Draft a reply to an email"""
    account_email = email_data.get('account_email')
    email_id = email_data.get('email_id')
    
    if not account_email or not email_id:
        raise HTTPException(status_code=400, detail="Missing account_email or email_id")
    
    db = SessionLocal()
    acc = db.query(GmailAccount).filter(GmailAccount.email == account_email).first()
    db.close()
    
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")
    
    service = get_gmail_service(acc.token_json)
    msg = service.users().messages().get(userId='me', id=email_id, format='full').execute()
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
             f"Here is an example of the tone that I normally use for a formal email: " \
             f"Dear Daniel, Thank you for the opportunity to interview with" \
             f"Cogo Labs today. It was great talking to you and I enjoyed learning more about" \
             f"Cogos Labs' initiatives and the internship program. I am really excited about the" \
             f"possibility of studying AI and developing a system that can ultimately be" \
             f"implemented in the real-world. Thank you for your time and consideration." \
             f"Here is an example of the tone that I normally use for a less formal email: " \
             f"Hi! Thank you for reaching out! I really enjoyed working at the PCF camp. Unfortunately, I won’t be able to return this summer as I’m interning,\n" \
             f"but I wanted to recommend my younger sister, Kate, if you're looking for coaches. Kate is a" \
             f"rising sophomore at North Andover High School, a varsity basketball player, and great with kids - I think she would be an excellent coach!" \
             f"Thank you again! " \
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
