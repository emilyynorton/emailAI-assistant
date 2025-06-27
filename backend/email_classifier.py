import os
import json
import base64
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import openai
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# GENERAL IDEA:
# - fetches 20 most recent read and starred, read and deleted, read and sitting in inbox emails
# - uses OpenAI embeddings to convert those to vectors (X = list of embeddings, Y = classification as trash/important)
# - uses sklearn to train a model on those embeddings
    # - splits set into training and testing set
    # - uses RandomForestClassifier to combine the outputs of the decision trees to make final decision
# - uses trained model to predict category for new emails
# - saves model to disk for future use
    # - won't make a new model if one already exists and is less than 24 hours old

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Categories for email classification
EMAIL_CATEGORIES = {
    "important": "Important email that requires attention",
    "regular": "Normal email that can be read later",
    "trash": "Spam, promotional, or irrelevant email that can be deleted"
}

class EmailClassifier:
    """Class to classify emails into important, regular, or trash"""
    
    def __init__(self):
        # Default model path template - will be formatted with account email
        self.model_path_template = "email_classification_model_{}.pkl"
        self.embedding_cache = {}
        self.vectorizer = None
        # Dictionary to store classifiers for different accounts
        self.classifiers = {}
        
        # Try to load embedding cache if it exists
        cache_path = "email_embedding_cache.pkl"
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            print(f"Error loading embedding cache: {str(e)}")
            # Continue with empty cache
            self.embedding_cache = {}
    
    def _get_model_path(self, account_email):
        """Get the model path for a specific account"""
        # Normalize email address to use as filename (replace @ and . with _)
        safe_email = account_email.replace('@', '_at_').replace('.', '_dot_')
        return self.model_path_template.format(safe_email)
        
    def get_gmail_service(self, token_json):
        """Create a Gmail API service instance."""
        SCOPES = [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.modify"
        ]
        creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
        return build('gmail', 'v1', credentials=creds)
    
    def fetch_training_data(self, token_json: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Fetch starred emails (important) and trash emails for training data
        
        Returns:
            Tuple of (important_emails, regular_emails, trash_emails)
        """
        service = self.get_gmail_service(token_json)
        
        # Use much smaller sample sizes for faster training
        # For production, these numbers could be increased
        MAX_IMPORTANT = 20
        MAX_TRASH = 20
        MAX_REGULAR = 20
        
        print(f"Fetching training data (max {MAX_IMPORTANT} important, {MAX_REGULAR} regular, {MAX_TRASH} trash emails)...")
        
        # Get starred/important emails
        important_emails = []
        response = service.users().messages().list(
            userId='me', q='is:starred is:read', maxResults=MAX_IMPORTANT
        ).execute()
        messages = response.get('messages', [])
        
        for m in messages:
            try:
                msg = service.users().messages().get(
                    userId='me', id=m['id'], format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                important_emails.append(self._process_email_lightweight(msg, 'important'))
                if len(important_emails) >= MAX_IMPORTANT:
                    break
            except Exception as e:
                print(f"Error processing starred message {m['id']}: {str(e)}")
        
        # Get trash emails
        trash_emails = []
        response = service.users().messages().list(
            userId='me', q='in:trash is:read', maxResults=MAX_TRASH
        ).execute()
        messages = response.get('messages', [])
        
        for m in messages:
            try:
                msg = service.users().messages().get(
                    userId='me', id=m['id'], format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                trash_emails.append(self._process_email_lightweight(msg, 'trash'))
                if len(trash_emails) >= MAX_TRASH:
                    break
            except Exception as e:
                print(f"Error processing trash message {m['id']}: {str(e)}")
        
        # Get regular emails (not starred, not in trash, not in spam)
        regular_emails = []
        response = service.users().messages().list(
            userId='me', q='NOT in:trash NOT in:spam -is:starred is:read', maxResults=MAX_REGULAR
        ).execute()
        messages = response.get('messages', [])
        
        for m in messages:
            try:
                msg = service.users().messages().get(
                    userId='me', id=m['id'], format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                regular_emails.append(self._process_email_lightweight(msg, 'regular'))
                if len(regular_emails) >= MAX_REGULAR:
                    break
            except Exception as e:
                print(f"Error processing regular message {m['id']}: {str(e)}")
        
        print(f"Fetched {len(important_emails)} important, {len(regular_emails)} regular, {len(trash_emails)} trash emails for training")
        return (important_emails, regular_emails, trash_emails)
    
    def _process_email(self, email: Dict, label: str) -> Dict:
        """Extract relevant features from email and assign label"""
        payload = email.get('payload', {})
        headers = {h['name']: h['value'] for h in payload.get('headers', [])}
        
        # Extract email body
        body = self._get_email_body(payload)
        
        return {
            'id': email['id'],
            'subject': headers.get('Subject', ''),
            'from': headers.get('From', ''),
            'date': headers.get('Date', ''),
            'snippet': email.get('snippet', ''),
            'body': body,
            'label': label
        }
        
    def _process_email_lightweight(self, email: Dict, label: str) -> Dict:
        """Extract minimal features from email for faster training"""
        headers = {}
        for header in email.get('payload', {}).get('headers', []):
            headers[header['name']] = header['value']
        
        return {
            'id': email['id'],
            'subject': headers.get('Subject', ''),
            'from': headers.get('From', ''),
            'date': headers.get('Date', ''),
            'snippet': email.get('snippet', ''),
            'label': label
        }
    
    def _get_email_body(self, payload: Dict) -> str:
        """Extract the email body from the payload"""
        body = ''
        
        # Check if the message has parts
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part.get('body', {}):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        # If no parts or no text/plain part found, try the body directly
        elif 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
            
        return body
    
    def train_model(self, account_email: str, token_json: str) -> None:
        """
        Train an LLM model to classify emails into important, regular, or trash
        for a specific account
        """
        # Get account-specific model path
        model_path = self._get_model_path(account_email)
        
        # Check if model already exists and is less than 24 hours old
        import os.path
        import time
        
        try:
            if os.path.exists(model_path):
                model_age = time.time() - os.path.getmtime(model_path)
                if model_age < 24 * 3600:  # 24 hours in seconds
                    print(f"Using existing model for {account_email} that's less than 24 hours old")
                    # Load the existing model
                    with open(model_path, 'rb') as f:
                        self.classifiers[account_email] = pickle.load(f)
                    return
        except Exception as e:
            print(f"Error checking model age for {account_email}: {str(e)}")
        
        # Fetch training data
        important_emails, regular_emails, trash_emails = self.fetch_training_data(token_json)
        all_emails = important_emails + regular_emails + trash_emails
        
        if len(all_emails) < 6:
            print("Not enough training data, need at least 6 emails")
            return
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(all_emails)
        
        # Combine features for training - subject and sender are most important
        df['text'] = df['subject'] + ' ' + df['from'] + ' ' + df['snippet']
        
        # Use OpenAI embeddings for better results
        print("Generating embeddings for emails...")
        X = self._get_embeddings(df['text'].tolist())
        y = df['label'].tolist()
        
        # Split data - use stratify to ensure each class is represented
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails (e.g., only one class), do without it
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train model (using RandomForest as it works well with embeddings)
        from sklearn.ensemble import RandomForestClassifier
        
        print("Training classifier model...")
        classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        accuracy = classifier.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save model for this specific account
        self.classifiers[account_email] = classifier
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
            
        # Also save the embedding cache to speed up future classifications
        cache_path = "email_embedding_cache.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"Embedding cache saved to {cache_path}")
        except Exception as e:
            print(f"Error saving embedding cache: {str(e)}")
            
        print(f"Model for {account_email} saved to {model_path}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get OpenAI embeddings for texts"""
        embeddings = []
        batch_size = 5  # Process in smaller batches for better responsiveness
        cache_hits = 0
        api_calls = 0
        
        # Create a cache lookup for just this batch to avoid duplicate API calls
        to_process = []
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
                cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                to_process.append((i, text))  # (index, text)
        
        if to_process:
            # Process the texts that weren't in the cache
            for i in range(0, len(to_process), batch_size):
                batch = to_process[i:i+batch_size]
                batch_indices = [item[0] for item in batch]
                batch_texts = [item[1] for item in batch]
                
                try:
                    # Use text-embedding-3-small for cost efficiency
                    api_calls += 1
                    response = openai.embeddings.create(
                        input=batch_texts,
                        model="text-embedding-3-small"
                    )
                    
                    # Fill in the embeddings and update cache
                    for j, idx in enumerate(batch_indices):
                        embedding = response.data[j].embedding
                        self.embedding_cache[texts[idx]] = embedding
                        embeddings[idx] = embedding
                        
                except Exception as e:
                    print(f"Error getting embeddings batch: {str(e)}")
                    # Fallback to zero vector if error
                    for idx in batch_indices:
                        embeddings[idx] = [0.0] * 1536  # Size of OpenAI embeddings
        
        if cache_hits > 0 or api_calls > 0:
            print(f"Embedding stats: {cache_hits} cache hits, {api_calls} API calls")
            
        # Make sure there are no None values
        for i, emb in enumerate(embeddings):
            if emb is None:
                embeddings[i] = [0.0] * 1536  # Fallback if something went wrong
                
        return embeddings

    def classify_email(self, email_data: Dict) -> str:
        """
        Classify an email as important, regular, or trash
        
        Args:
            email_data: Dict containing 'subject', 'from', 'snippet', and 'account'
        
        Returns:
            Category as string: 'important', 'regular', or 'trash'
        """
        account_email = email_data.get('account')
        if not account_email:
            print("No account email provided for classification")
            return 'regular'  # Default category
        
        # Load model for this account if not loaded
        if account_email not in self.classifiers:
            model_path = self._get_model_path(account_email)
            try:
                with open(model_path, 'rb') as f:
                    self.classifiers[account_email] = pickle.load(f)
                    print(f"Loaded model for {account_email}")
            except FileNotFoundError:
                print(f"No trained model found for {account_email}. Using LLM for classification.")
                return self._llm_classify(email_data)
            except Exception as e:
                print(f"Error loading model for {account_email}: {str(e)}")
                return self._llm_classify(email_data)
        
        # Extract text features
        text = f"{email_data['subject']} {email_data['from']} {email_data['snippet']}"
        
        # Get embedding
        embedding = self._get_embeddings([text])[0]
        
        # Make prediction using this account's classifier
        classifier = self.classifiers[account_email]
        prediction = classifier.predict([embedding])[0]
        prediction_proba = max(classifier.predict_proba([embedding])[0])
        
        # If confidence is low, use LLM for classification
        if prediction_proba < 0.7:
            return self._llm_classify(email_data)
        
        return prediction
    
    def _llm_classify(self, email_data: Dict) -> str:
        """Use LLM to classify an email when confidence is low"""
        prompt = f"""
        Classify the following email into one of these categories:
        - important: Important email that requires attention
        - regular: Normal email that can be read later
        - trash: Spam, promotional, or irrelevant email that can be deleted
        
        Email:
        From: {email_data.get('from', '')}
        Subject: {email_data.get('subject', '')}
        Snippet: {email_data.get('snippet', '')}
        
        Classification (important/regular/trash):
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3,
            )
            result = response.choices[0].message.content.strip().lower()
            
            # Extract just the label
            for category in EMAIL_CATEGORIES:
                if category in result:
                    return category
                    
            # Default to regular if we couldn't extract a clear category
            return "regular"
            
        except Exception as e:
            print(f"Error using LLM for classification: {str(e)}")
            return "regular"  # Default if API fails
