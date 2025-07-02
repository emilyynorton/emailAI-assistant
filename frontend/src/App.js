import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';
import config from './config';

function App() {
  const [accounts, setAccounts] = useState([]);
  const [emails, setEmails] = useState([]);
  const [loading, setLoading] = useState(false);
  const [drafts, setDrafts] = useState({});
  const [draftLoading, setDraftLoading] = useState({});
  const [deleteLoading, setDeleteLoading] = useState({});
  const [recoverLoading, setRecoverLoading] = useState({});
  const [categoryLoading, setCategoryLoading] = useState({});
  const [error, setError] = useState(null);
  const [userName, setUserName] = useState(localStorage.getItem('userName') || '');
  const [isEditingName, setIsEditingName] = useState(false);
  const [activeCategory, setActiveCategory] = useState('all'); // 'all', 'important', 'regular', 'trash'

  // Fetch connected accounts
  useEffect(() => {
    axios.get(`${config.API_URL}/accounts`)
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
    axios.get(`${config.API_URL}/emails/today`)
      .then(res => {
        setEmails(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch emails.');
        setLoading(false);
      });
  }, []);

  // Filter emails based on active category
  const filteredEmails = emails.filter(email => {
    if (activeCategory === 'all') return true;
    return email.category === activeCategory;
  });

  // Count emails by category
  const emailCounts = {
    all: emails.length,
    important: emails.filter(email => email.category === 'important').length,
    regular: emails.filter(email => email.category === 'regular').length,
    trash: emails.filter(email => email.category === 'trash').length
  };
  
  // Periodically refresh emails to get updated classifications
  useEffect(() => {
    // Set up a refresh interval (every 2 minutes)
    const refreshInterval = setInterval(() => {
      if (accounts.length > 0) {
        axios.get(`${config.API_URL}/emails/today`)
          .then(res => {
            setEmails(res.data);
          })
          .catch(err => {
            console.error('Failed to refresh emails:', err);
          });
      }
    }, 120000); // 2 minutes
    
    // Clean up the interval on unmount
    return () => clearInterval(refreshInterval);
  }, [accounts]);

  // Handle updating email category
  const handleUpdateCategory = async (email, newCategory) => {
    setCategoryLoading(prev => ({ ...prev, [email.id]: true }));
    try {
      await axios.post(`${config.API_URL}/emails/update_category`, {
        email_id: email.id,
        account_email: email.account,
        category: newCategory
      });
      
      // Update local state
      setEmails(prevEmails => 
        prevEmails.map(e => 
          e.id === email.id ? { ...e, category: newCategory } : e
        )
      );
    } catch (error) {
      setError(`Failed to update category: ${error.response?.data?.detail || error.message}`);
    } finally {
      setCategoryLoading(prev => ({ ...prev, [email.id]: false }));
    }
  };

  const handleDraft = async (email) => {
    setDraftLoading(draft => ({ ...draft, [email.id]: true }));
    setDrafts(draft => ({ ...draft, [email.id]: null }));
    try {
      const res = await axios.post(`${config.API_URL}/emails/draft_reply`, {
        account_email: email.account,
        email_id: email.id,
        thread_id: email.threadId,
        sender_name: userName // Send the user name to the backend
      });
      setDrafts(draft => ({ ...draft, [email.id]: res.data.draft }));
    } catch (e) {
      setDrafts(draft => ({ ...draft, [email.id]: 'Failed to generate draft.' }));
    }
    setDraftLoading(draft => ({ ...draft, [email.id]: false }));
  };

  const handleDelete = async (email) => {
    setDeleteLoading((prev) => ({ ...prev, [email.id]: true }));
    try {
      await axios.post(`${config.API_URL}/emails/delete`, {
        email_id: email.id,
        account_email: email.account
      });
      
      // Instead of removing the email, update its category to 'trash'
      // This keeps it visible in the trash tab
      setEmails(emails.map(e => {
        if (e.id === email.id) {
          return { ...e, category: 'trash' };
        }
        return e;
      }));
      
    } catch (error) {
      setError('Failed to delete email: ' + error.message);
    } finally {
      setDeleteLoading((prev) => ({ ...prev, [email.id]: false }));
    }
  };

  const handleRecover = async (email) => {
    setRecoverLoading((prev) => ({ ...prev, [email.id]: true }));
    try {
      await axios.post(`${config.API_URL}/emails/recover`, {
        email_id: email.id,
        account_email: email.account
      });
      
      // Update the email category in the UI
      setEmails(emails.map(e => {
        if (e.id === email.id) {
          return { ...e, category: 'regular', auto_trashed: false };
        }
        return e;
      }));
      
    } catch (error) {
      setError('Failed to recover email: ' + error.message);
    } finally {
      setRecoverLoading((prev) => ({ ...prev, [email.id]: false }));
    }
  };

  const handleConnectAccount = () => {
    window.location.href = `${config.API_URL}/auth/google`;
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
        <p className="hint-text">
          Emails are automatically sorted based on your past behavior (starred emails are used to inform "important" emails and trash emails are used to inform "trash").
        </p>
        
        <div className="category-tabs">
          <button 
            className={`tab-button ${activeCategory === 'all' ? 'active' : ''}`}
            onClick={() => setActiveCategory('all')}
          >
            All ({emailCounts.all})
          </button>
          <button 
            className={`tab-button ${activeCategory === 'important' ? 'active' : ''}`}
            onClick={() => setActiveCategory('important')}
          >
            Important ({emailCounts.important})
          </button>
          <button 
            className={`tab-button ${activeCategory === 'regular' ? 'active' : ''}`}
            onClick={() => setActiveCategory('regular')}
          >
            Regular ({emailCounts.regular})
          </button>
          <button 
            className={`tab-button ${activeCategory === 'trash' ? 'active' : ''}`}
            onClick={() => setActiveCategory('trash')}
          >
            Trash ({emailCounts.trash})
          </button>
        </div>
        
        {loading && <p>Loading emails...</p>}
        {error && <div className="error-message">{error}</div>}
        {!loading && filteredEmails.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">✓</div>
            <p>{emails.length === 0 ? 'No missed emails found for today!' : `No ${activeCategory} emails found.`}</p>
          </div>
        )}
        <ul className="email-list">
        {filteredEmails.map(email => (
          <li key={email.id} className={`email-item email-category-${email.category || 'regular'}`}>
            <div className="email-account">
              {email.account}
              <span className={`category-badge category-${email.category || 'regular'}`}>
                {email.category || 'regular'}
              </span>
            </div>
            <div className="email-header">
              <div className="email-from"><b>From:</b> {email.from}</div>
              <div className="email-subject">{email.subject}</div>
              <div className="email-date">{email.date}</div>
            </div>
            <div className="email-snippet">{email.snippet}</div>
            {email.auto_trashed && (
              <div className="auto-trashed-notice">
                This email was automatically moved to trash based on your preferences
              </div>
            )}
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
              
              <div className="category-actions">
                <label>Category:</label>
                <select 
                  value={email.category || 'regular'} 
                  onChange={(e) => handleUpdateCategory(email, e.target.value)}
                  disabled={categoryLoading[email.id]}
                >
                  <option value="important">Important</option>
                  <option value="regular">Regular</option>
                  <option value="trash">Trash</option>
                </select>
                {categoryLoading[email.id] && <span className="loading-spinner small"></span>}
              </div>
              
              {email.category === 'trash' && (
                <button 
                  className="button button-recover"
                  onClick={() => handleRecover(email)}
                  disabled={recoverLoading[email.id]}
                >
                  {recoverLoading[email.id] ? (
                    <><span className="loading-spinner"></span> Recovering...</>
                  ) : (
                    'Recover Email'
                  )}
                </button>
              )}
              
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
