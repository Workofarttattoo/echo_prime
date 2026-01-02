import React, { useState, useEffect } from 'react';

const ConsciousnessControls = ({ onConsciousnessCommand, consciousnessMetrics }) => {
  const [analysisType, setAnalysisType] = useState('phi');
  const [systemState, setSystemState] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState(null);

  const analysisTypes = {
    phi: {
      name: 'Phi Calculation',
      description: 'Measure integrated information (consciousness level)',
      icon: 'Φ'
    },
    repertoire: {
      name: 'Cause-Effect Repertoire',
      description: 'Analyze causal relationships in system state',
      icon: '🔗'
    },
    complexity: {
      name: 'Complexity Analysis',
      description: 'Measure information complexity and integration',
      icon: '🧩'
    },
    partition: {
      name: 'Partition Analysis',
      description: 'Find minimum information partitions',
      icon: '✂️'
    }
  };

  const generateRandomState = () => {
    const size = Math.floor(Math.random() * 8) + 4; // 4-12 elements
    const state = Array.from({length: size}, () => Math.random());
    setSystemState(JSON.stringify(state));
  };

  const runAnalysis = async () => {
    if (!systemState.trim()) return;

    setIsAnalyzing(true);
    try {
      let state;
      try {
        state = JSON.parse(systemState);
      } catch {
        state = systemState.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
      }

      const result = await onConsciousnessCommand(analysisType, {
        system_state: state
      });

      setLastAnalysis({
        type: analysisType,
        result,
        input: state,
        timestamp: new Date().toLocaleTimeString()
      });
    } catch (error) {
      console.error('Consciousness analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConsciousnessLevel = (phi) => {
    if (phi < 0.1) return { level: 'minimal', color: '#666' };
    if (phi < 0.5) return { level: 'basic', color: '#ffa500' };
    if (phi < 1.0) return { level: 'moderate', color: '#ffff00' };
    if (phi < 2.0) return { level: 'advanced', color: '#00ff88' };
    return { level: 'high', color: '#00f2ff' };
  };

  const consciousnessLevel = consciousnessMetrics?.phi ?
    getConsciousnessLevel(consciousnessMetrics.phi) : null;

  return (
    <div className="glass-panel">
      <div className="panel-header">🧠 CONSCIOUSNESS MONITOR</div>

      {/* Current Metrics */}
      <div style={{ marginBottom: '15px', fontSize: '11px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
          <span style={{ color: '#acc-secondary' }}>Phi (Φ):</span>
          <span style={{ color: consciousnessLevel?.color || '#fff' }}>
            {consciousnessMetrics?.phi?.toFixed(4) || 'N/A'}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
          <span style={{ color: '#acc-secondary' }}>Level:</span>
          <span style={{ color: consciousnessLevel?.color || '#fff' }}>
            {consciousnessLevel?.level || 'unknown'}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
          <span style={{ color: '#acc-secondary' }}>Cause Complexity:</span>
          <span>{consciousnessMetrics?.cause_complexity?.toFixed(2) || 'N/A'}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#acc-secondary' }}>Effect Complexity:</span>
          <span>{consciousnessMetrics?.effect_complexity?.toFixed(2) || 'N/A'}</span>
        </div>
      </div>

      {/* Analysis Type Selector */}
      <div style={{ marginBottom: '10px' }}>
        <select
          value={analysisType}
          onChange={(e) => setAnalysisType(e.target.value)}
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
          {Object.entries(analysisTypes).map(([key, config]) => (
            <option key={key} value={key}>
              {config.icon} {config.name}
            </option>
          ))}
        </select>
      </div>

      {/* Analysis Description */}
      <div style={{
        fontSize: '10px',
        color: '#acc-secondary',
        marginBottom: '10px',
        minHeight: '20px'
      }}>
        {analysisTypes[analysisType].description}
      </div>

      {/* System State Input */}
      <div style={{ marginBottom: '10px' }}>
        <textarea
          value={systemState}
          onChange={(e) => setSystemState(e.target.value)}
          placeholder="Enter system state as JSON array [0.1, 0.5, 0.8, ...] or comma-separated values"
          style={{
            width: '100%',
            height: '40px',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            color: '#fff',
            padding: '5px',
            fontSize: '9px',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px',
            resize: 'none'
          }}
        />
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
        <button
          onClick={generateRandomState}
          style={{
            flex: 1,
            background: 'rgba(138,43,226,0.2)',
            border: '1px solid rgba(138,43,226,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '9px',
            cursor: 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          🎲 RANDOM STATE
        </button>

        <button
          onClick={runAnalysis}
          disabled={isAnalyzing || !systemState.trim()}
          style={{
            flex: 1,
            background: isAnalyzing ? 'rgba(255,165,0,0.3)' : 'rgba(255,69,0,0.2)',
            border: '1px solid rgba(255,69,0,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '9px',
            cursor: (isAnalyzing || !systemState.trim()) ? 'not-allowed' : 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          {isAnalyzing ? 'ANALYZING...' : '🔬 ANALYZE'}
        </button>
      </div>

      {/* Last Analysis Display */}
      {lastAnalysis && (
        <div style={{
          fontSize: '10px',
          color: '#acc-secondary',
          padding: '5px',
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '3px',
          maxHeight: '80px',
          overflowY: 'auto'
        }}>
          <div>Last: {analysisTypes[lastAnalysis.type].icon} {lastAnalysis.timestamp}</div>
          {lastAnalysis.result?.phi && (
            <div style={{ fontSize: '9px', marginTop: '2px' }}>
              Phi: {lastAnalysis.result.phi.toFixed(4)}
              <span style={{
                color: getConsciousnessLevel(lastAnalysis.result.phi).color,
                marginLeft: '5px'
              }}>
                ({getConsciousnessLevel(lastAnalysis.result.phi).level})
              </span>
            </div>
          )}
          {lastAnalysis.result?.consciousness_level && (
            <div style={{ fontSize: '9px', marginTop: '2px' }}>
              Level: {lastAnalysis.result.consciousness_level}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ConsciousnessControls;


