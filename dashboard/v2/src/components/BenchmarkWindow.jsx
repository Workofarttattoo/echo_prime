import React from 'react';

const BenchmarkWindow = ({ benchmarks }) => {
    if (!benchmarks || Object.keys(benchmarks).length === 0) return null;

    return (
        <div className="glass-panel" style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <div className="panel-header" style={{ color: '#00f2ff', borderBottom: '1px solid rgba(0, 242, 255, 0.2)' }}>
                AGI REASONING BENCHMARKS
            </div>
            <div style={{ padding: '10px 0' }}>
                {Object.entries(benchmarks).map(([name, stats]) => (
                    <div key={name} style={{ marginBottom: '15px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '5px' }}>
                            <span style={{ fontWeight: 'bold', color: '#fff' }}>{name}</span>
                            <span style={{ color: '#00ff88' }}>{stats.passed}/{stats.current} | {stats.progress}</span>
                        </div>
                        <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
                            <div
                                style={{
                                    width: stats.progress,
                                    height: '100%',
                                    background: 'linear-gradient(90deg, #00f2ff, #00ff88)',
                                    borderRadius: '2px',
                                    boxShadow: '0 0 10px #00f2ff'
                                }}
                            />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default BenchmarkWindow;
