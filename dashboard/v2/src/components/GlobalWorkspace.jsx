import React from 'react';

const GlobalWorkspace = ({ modules = [] }) => {
    // Mock modules if not provided
    const activeModules = modules.length > 0 ? modules : [
        { name: 'SENSORY', state: 'ACTIVE', load: 45 },
        { name: 'ATTENTION', state: 'RESONATING', load: 88 },
        { name: 'MEMORY', state: 'RETRIEVING', load: 32 },
        { name: 'REASONING', state: 'PROCESSING', load: 67 },
        { name: 'SAFETY', state: 'MONITORING', load: 12 }
    ];

    return (
        <div className="panel" style={{ marginTop: '20px' }}>
            <h2>GLOBAL WORKSPACE ACTIVITY</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '15px', marginTop: '10px' }}>
                {activeModules.map((m, i) => (
                    <div key={i} style={{
                        background: 'rgba(255,255,255,0.05)',
                        padding: '12px',
                        borderRadius: '8px',
                        border: `1px solid ${m.load > 80 ? 'var(--acc-primary)' : 'rgba(255,255,255,0.1)'}`
                    }}>
                        <div style={{ fontSize: '10px', color: '#888', marginBottom: '8px' }}>{m.name}</div>
                        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '5px' }}>
                            <div style={{
                                height: '4px',
                                flex: 1,
                                background: '#222',
                                borderRadius: '2px',
                                overflow: 'hidden'
                            }}>
                                <div style={{
                                    width: `${m.load}%`,
                                    height: '100%',
                                    background: m.load > 80 ? 'var(--acc-primary)' : 'var(--acc-secondary)',
                                    boxShadow: m.load > 80 ? '0 0 10px var(--acc-primary)' : 'none'
                                }} />
                            </div>
                            <div style={{ fontSize: '10px', fontFamily: 'JetBrains Mono' }}>{m.load}%</div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default GlobalWorkspace;
