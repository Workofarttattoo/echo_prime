import React, { useState } from 'react';

function TextInput({ onSend, apiBaseUrl = 'http://127.0.0.1:8000' }) {
  const [inputText, setInputText] = useState('');
  const [isSending, setIsSending] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isSending) return;

    setIsSending(true);
    try {
      // Send text to the backend API
      const response = await fetch(`${apiBaseUrl}/text-input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText.trim() })
      });

      if (response.ok) {
        if (onSend) {
          onSend(inputText.trim());
        }
        setInputText(''); // Clear input after sending
      } else {
        console.error('Failed to send text:', response.statusText);
      }
    } catch (error) {
      console.error('Error sending text:', error);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="glass-panel" style={{ marginTop: '10px' }}>
      <div className="panel-header">TEXT INPUT</div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your command or question here... (Press Enter to send, Shift+Enter for new line)"
          style={{
            width: '100%',
            minHeight: '80px',
            padding: '10px',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            resize: 'vertical',
            outline: 'none',
            transition: 'border-color 0.3s'
          }}
          onFocus={(e) => e.target.style.borderColor = 'rgba(0, 242, 255, 0.5)'}
          onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)'}
          disabled={isSending}
        />
        <button
          type="submit"
          disabled={!inputText.trim() || isSending}
          style={{
            padding: '8px 16px',
            background: isSending || !inputText.trim() 
              ? 'rgba(255, 255, 255, 0.1)' 
              : 'rgba(0, 242, 255, 0.3)',
            border: `1px solid ${isSending || !inputText.trim() 
              ? 'rgba(255, 255, 255, 0.2)' 
              : 'rgba(0, 242, 255, 0.5)'}`,
            color: '#fff',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            cursor: isSending || !inputText.trim() ? 'not-allowed' : 'pointer',
            borderRadius: '4px',
            transition: 'all 0.3s',
            opacity: isSending || !inputText.trim() ? 0.5 : 1
          }}
          onMouseEnter={(e) => {
            if (!isSending && inputText.trim()) {
              e.target.style.background = 'rgba(0, 242, 255, 0.4)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isSending && inputText.trim()) {
              e.target.style.background = 'rgba(0, 242, 255, 0.3)';
            }
          }}
        >
          {isSending ? 'SENDING...' : 'SEND'}
        </button>
      </form>
      <div style={{ 
        fontSize: '10px', 
        color: 'rgba(255, 255, 255, 0.5)', 
        marginTop: '5px',
        fontFamily: 'var(--font-mono)'
      }}>
        Enter to send • Shift+Enter for new line
      </div>
    </div>
  );
}

export default TextInput;

