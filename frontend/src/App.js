import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [accounts, setAccounts] = useState([]);
  const [emails, setEmails] = useState([]);
  const [loading, setLoading] = useState(false);
  const [drafts, setDrafts] = useState({});
  const [draftLoading, setDraftLoading] = useState({});
  const [deleteLoading, setDeleteLoading] = useState({});
  const [error, setError] = useState(null);
  const [userName, setUserName] = useState(localStorage.getItem('userName') || '');
  const [isEditingName, setIsEditingName] = useState(false);

  // Fetch connected accounts
  useEffect(() => {
    axios.get('http://localhost:8000/accounts')
      .then(res => setAccounts(res.data))
      .catch(() => setAccounts([]));
  }, []);
  
  // Save userName to localStorage when it changes
  useEffect(() => {
    if (userName) {
      localStorage.setItem('userName', userName);
    }
  }, [userName]);

  // Fetch today's unreplied emails
  useEffect(() => {
    setLoading(true);
    axios.get('http://localhost:8000/emails/today')
      .then(res => {
        setEmails(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch emails.');
        setLoading(false);
      });
  }, []);

  const handleDraft = async (email) => {
    setDraftLoading(draft => ({ ...draft, [email.id]: true }));
    setDrafts(draft => ({ ...draft, [email.id]: null }));
    try {
      const res = await axios.post('http://localhost:8000/emails/draft_reply', {
        account: email.account,
        id: email.id,
        threadId: email.threadId,
        sender_name: userName // Send the user name to the backend
      });
      setDrafts(draft => ({ ...draft, [email.id]: res.data.draft }));
    } catch (e) {
      setDrafts(draft => ({ ...draft, [email.id]: 'Failed to generate draft.' }));
    }
    setDraftLoading(draft => ({ ...draft, [email.id]: false }));
  };

  const handleDelete = async (email) => {
    if (window.confirm(`Are you sure you want to delete this email from ${email.from}?`)) {
      setDeleteLoading(loading => ({ ...loading, [email.id]: true }));
      try {
        await axios.post('http://localhost:8000/emails/delete', {
          account: email.account,
          id: email.id
        });
        // Remove the email from the list
        setEmails(emails => emails.filter(e => e.id !== email.id));
      } catch (e) {
        setError(`Failed to delete email: ${e.response?.data?.detail || e.message}`);
      } finally {
        setDeleteLoading(loading => ({ ...loading, [email.id]: false }));
      }
    }
  };

  const handleConnectAccount = () => {
    window.location.href = 'http://localhost:8000/auth/google';
  };

  const handleSaveName = () => {
    setIsEditingName(false);
  };

  const handleNameChange = (e) => {
    setUserName(e.target.value);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Email AI Assistant</h1>
        <p className="app-subtitle">Manage your emails efficiently with AI assistance</p>
      </header>
      <div className="card">
        <h3 className="card-title">Your Name for Email Drafts</h3>
        {isEditingName ? (
          <div>
            <input 
              type="text" 
              value={userName} 
              onChange={handleNameChange} 
              placeholder="Enter your name"
              className="input-field"
            />
            <button className="button" onClick={handleSaveName}>Save</button>
          </div>
        ) : (
          <div>
            <span style={{ fontSize: '16px', marginRight: '10px' }}>
              {userName || 'No name set (emails will use your email username)'}
            </span>
            <button className="button" onClick={() => setIsEditingName(true)}>Edit</button>
          </div>
        )}
      </div>
      <div className="card">
        <h3 className="card-title">Connected Gmail Accounts</h3>
        {accounts.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📧</div>
            <p>No accounts connected yet.</p>
          </div>
        ) : (
          <ul style={{ paddingLeft: 18 }}>
            {accounts.map(acc => (
              <li key={acc.email}>{acc.email}</li>
            ))}
          </ul>
        )}
        <button className="button" onClick={handleConnectAccount}>
          Connect Gmail Account
        </button>
      </div>
      <div className="card">
        <h3 className="card-title">Today's Unreplied Emails</h3>
        {loading && <p>Loading emails...</p>}
        {error && <div className="error-message">{error}</div>}
        {!loading && emails.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">✓</div>
            <p>No missed emails found for today!</p>
          </div>
        )}
        <ul className="email-list">
        {emails.map(email => (
          <li key={email.id} className="email-item">
            <div className="email-account">{email.account}</div>
            <div className="email-header">
              <div className="email-from"><b>From:</b> {email.from}</div>
              <div className="email-subject">{email.subject}</div>
              <div className="email-date">{email.date}</div>
            </div>
            <div className="email-snippet">{email.snippet}</div>
            <div className="button-container">
              <button 
                className="button"
                onClick={() => handleDraft(email)} 
                disabled={draftLoading[email.id]}
              >
                {draftLoading[email.id] ? (
                  <><span className="loading-spinner"></span> Drafting...</>
                ) : (
                  'Draft Reply'
                )}
              </button>
              <button 
                className="button button-danger"
                onClick={() => handleDelete(email)}
                disabled={deleteLoading[email.id]}
              >
                {deleteLoading[email.id] ? (
                  <><span className="loading-spinner"></span> Deleting...</>
                ) : (
                  'Delete Email'
                )}
              </button>
            </div>
            {drafts[email.id] && (
              <div className="draft-container">
                <div className="draft-title">Suggested Draft:</div>
                <div className="draft-content">{drafts[email.id]}</div>
              </div>
            )}
          </li>
        ))}
      </ul>
      </div>
    </div>
  );
}

export default App;
