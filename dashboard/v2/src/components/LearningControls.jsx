import React, { useState, useEffect } from 'react';

const LearningControls = ({ onFeedbackSubmit, learningStats }) => {
  const [feedbackType, setFeedbackType] = useState('performance');
  const [feedbackContent, setFeedbackContent] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastFeedback, setLastFeedback] = useState(null);

  const feedbackTypes = {
    performance: {
      label: 'Performance Issue',
      placeholder: 'Describe the performance problem...',
      icon: '⚡'
    },
    correction: {
      label: 'Correction Needed',
      placeholder: 'What should it have done instead?',
      icon: '🔧'
    },
    preference: {
      label: 'User Preference',
      placeholder: 'How would you like it to behave?',
      icon: '👤'
    },
    success: {
      label: 'Success Feedback',
      placeholder: 'What worked well?',
      icon: '✅'
    },
    error: {
      label: 'Error Report',
      placeholder: 'Describe the error...',
      icon: '❌'
    }
  };

  const handleSubmit = async () => {
    if (!feedbackContent.trim()) return;

    setIsSubmitting(true);
    try {
      const feedbackId = await onFeedbackSubmit(feedbackType, {
        content: feedbackContent,
        timestamp: Date.now(),
        context: 'dashboard_user_input'
      });

      setLastFeedback({
        id: feedbackId,
        type: feedbackType,
        content: feedbackContent,
        timestamp: new Date().toLocaleTimeString()
      });

      setFeedbackContent('');
    } catch (error) {
      console.error('Feedback submission failed:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const forceLearningCycle = async () => {
    try {
      await fetch('http://127.0.0.1:8000/force_learning_cycle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (error) {
      console.error('Force learning cycle failed:', error);
    }
  };

  return (
    <div className="glass-panel">
      <div className="panel-header">🧠 CONTINUOUS LEARNING</div>

      {/* Learning Stats */}
      <div style={{ marginBottom: '15px', fontSize: '11px', color: '#acc-secondary' }}>
        <div>Cycles: {learningStats?.total_cycles || 0}</div>
        <div>Feedback: {learningStats?.feedback_stats?.total_feedback || 0}</div>
        <div>Adaptations: {learningStats?.adaptation_stats?.successful_adaptations || 0}</div>
      </div>

      {/* Feedback Type Selector */}
      <div style={{ marginBottom: '10px' }}>
        <select
          value={feedbackType}
          onChange={(e) => setFeedbackType(e.target.value)}
          style={{
            width: '100%',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            color: '#fff',
            padding: '5px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          {Object.entries(feedbackTypes).map(([key, config]) => (
            <option key={key} value={key}>
              {config.icon} {config.label}
            </option>
          ))}
        </select>
      </div>

      {/* Feedback Input */}
      <div style={{ marginBottom: '10px' }}>
        <textarea
          value={feedbackContent}
          onChange={(e) => setFeedbackContent(e.target.value)}
          placeholder={feedbackTypes[feedbackType].placeholder}
          style={{
            width: '100%',
            height: '60px',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            color: '#fff',
            padding: '5px',
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px',
            resize: 'none'
          }}
        />
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
        <button
          onClick={handleSubmit}
          disabled={isSubmitting || !feedbackContent.trim()}
          style={{
            flex: 1,
            background: isSubmitting ? 'rgba(255,165,0,0.3)' : 'rgba(0,255,136,0.2)',
            border: '1px solid rgba(0,255,136,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '10px',
            cursor: isSubmitting ? 'not-allowed' : 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          {isSubmitting ? 'SUBMITTING...' : '📤 SUBMIT FEEDBACK'}
        </button>

        <button
          onClick={forceLearningCycle}
          style={{
            flex: 1,
            background: 'rgba(0,242,255,0.2)',
            border: '1px solid rgba(0,242,255,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '10px',
            cursor: 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          🔄 FORCE LEARN
        </button>
      </div>

      {/* Last Feedback Display */}
      {lastFeedback && (
        <div style={{
          fontSize: '10px',
          color: '#acc-secondary',
          padding: '5px',
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '3px'
        }}>
          <div>Last: {feedbackTypes[lastFeedback.type].icon} {lastFeedback.timestamp}</div>
          <div style={{ fontSize: '9px', marginTop: '2px' }}>
            {lastFeedback.content.length > 50
              ? lastFeedback.content.substring(0, 50) + '...'
              : lastFeedback.content
            }
          </div>
        </div>
      )}
    </div>
  );
};

export default LearningControls;


