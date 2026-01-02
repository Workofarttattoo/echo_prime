import React from 'react';

const SensoryWindow = ({
    activeVisual,
    audioLevel = 0,
    isListening = false
}) => {
    return (
        <div className="glass-panel" style={{ minHeight: '350px', display: 'flex', flexDirection: 'column' }}>
            <div className="panel-header">
                <span className="icon">👁️</span> SENSORY INPUT
            </div>

            {/* Vision Feed */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden', borderRadius: '4px', background: '#000' }}>
                {activeVisual ? (
                    <img
                        src={activeVisual}
                        alt="Visual Input"
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                ) : (
                    <div style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        height: '100%', color: '#333', fontSize: '12px'
                    }}>
                        NO SIGNAL
                    </div>
                )}
                <div style={{ position: 'absolute', bottom: 5, left: 5, fontSize: '10px', color: '#0f0' }}>
                    CAM_01: {activeVisual ? "ACTIVE" : "STANDBY"}
                </div>
            </div>

            {/* Audio Waveform (Simulated CSS) */}
            <div style={{ height: '60px', marginTop: '10px', background: '#111', borderRadius: '4px', padding: '5px', position: 'relative' }}>
                <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>AUDIO SPECTRUM</div>
                <div style={{ display: 'flex', alignItems: 'flex-end', height: '30px', gap: '2px' }}>
                    {[...Array(20)].map((_, i) => (
                        <div key={i} style={{
                            flex: 1,
                            background: isListening ? '#0f0' : '#444',
                            height: `${Math.random() * (isListening ? 100 : 20)}%`,
                            opacity: 0.7,
                            transition: 'height 0.1s ease'
                        }} />
                    ))}
                </div>
                {isListening && <div style={{ position: 'absolute', top: 5, right: 5, width: 8, height: 8, borderRadius: '50%', background: '#f00', boxShadow: '0 0 5px #f00' }} />}
            </div>
        </div>
    );
};

export default SensoryWindow;
