import React from 'react';

const ThoughtStream = ({ insight, actions = [] }) => {
    return (
        <div className="panel" style={{ flex: 1 }}>
            <h2>PREFRONTAL THOUGHT STREAM</h2>
            <div id="thought-stream">
                {!insight ? (
                    <p style={{ color: '#aaa' }}>{">"} WAITING FOR SENSORY INPUT...</p>
                ) : (
                    <>
                        <div className="thought-entry">
                            <div className="thought-meta">{">"} INITIALIZING COGNITIVE RETRIEVAL...</div>
                            <div className="thought-meta">{">"} ANALYZING ANALOGIES...</div>
                            <div className="thought-content" dangerouslySetInnerHTML={{ __html: insight.replace(/\n/g, '<br>') }} />
                        </div>

                        {actions && actions.length > 0 && (
                            <div className="action-card">
                                <div className="action-header">EXECUTING ACTIONS...</div>
                                {actions.map((action, i) => (
                                    <div key={i} className="thought-content">{">"} {action}</div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default ThoughtStream;
