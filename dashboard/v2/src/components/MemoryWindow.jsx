import React, { useEffect, useRef } from 'react';

const MemoryWindow = ({ thoughts = [], memories = [], semanticCount = 0 }) => {
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [thoughts]);

    return (
        <div className="glass-panel" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div className="panel-header">
                <span className="icon">🧠</span> COGNITIVE STREAM
            </div>

            <div style={{ padding: '0 10px 10px', fontSize: '11px', color: '#aaa', display: 'flex', justifyContent: 'space-between' }}>
                <span>SEMANTIC_NODES: {semanticCount}</span>
                <span>MEM_FRAGMENTS: {memories.length}</span>
            </div>

            <div className="scroll-content" ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
                {thoughts.map((t, i) => (
                    <div key={i} className="thought-item" style={{
                        marginBottom: '12px', paddingLeft: '12px', borderLeft: '2px solid var(--acc-primary)',
                        animation: 'fadeIn 0.5s ease',
                        background: 'rgba(255,255,255,0.02)',
                        padding: '10px',
                        borderRadius: '0 8px 8px 0'
                    }}>
                        <div style={{ color: 'var(--acc-primary)', fontSize: '10px', fontFamily: 'JetBrains Mono', marginBottom: '4px', opacity: 0.6 }}>
                            [{t.time}] COGNITIVE_INSIGHT
                        </div>
                        <div style={{ color: '#fff', lineHeight: '1.5', fontSize: '12px' }}>{t.text}</div>
                    </div>
                ))}
                {memories.map((m, i) => (
                    <div key={`mem-${i}`} style={{ marginTop: '5px', fontSize: '10px', color: '#888' }}>
                        [RECALL] {m.text || "Vector Fragment"}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MemoryWindow;
