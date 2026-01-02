import React, { useState, useEffect } from 'react';

const SelfModControls = ({ onSelfModCommand, selfModStatus }) => {
  const [improvementType, setImprovementType] = useState('performance');
  const [targetCode, setTargetCode] = useState('');
  const [description, setDescription] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const improvementTypes = {
    performance: {
      label: 'Performance Optimization',
      description: 'Optimize code for speed/memory efficiency',
      icon: '⚡'
    },
    quality: {
      label: 'Code Quality',
      description: 'Add type hints, docstrings, error handling',
      icon: '🧹'
    },
    functionality: {
      label: 'Feature Enhancement',
      description: 'Add new capabilities or improve existing ones',
      icon: '🔧'
    },
    security: {
      label: 'Security Hardening',
      description: 'Add security checks and safe practices',
      icon: '🔒'
    },
    architecture: {
      label: 'Architecture Refactor',
      description: 'Improve code structure and organization',
      icon: '🏗️'
    }
  };

  const startSelfModification = async () => {
    if (!targetCode.trim() || !description.trim()) return;

    setIsRunning(true);
    try {
      const result = await onSelfModCommand('improve', {
        code: targetCode,
        improvement_type: improvementType,
        description,
        target_file: 'dashboard_generated.py'
      });

      setLastResult({
        type: improvementType,
        result,
        timestamp: new Date().toLocaleTimeString()
      });
    } catch (error) {
      console.error('Self-modification failed:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const analyzeCode = async () => {
    if (!targetCode.trim()) return;

    try {
      const analysis = await onSelfModCommand('analyze', {
        code: targetCode
      });

      setLastResult({
        type: 'analysis',
        result: analysis,
        timestamp: new Date().toLocaleTimeString()
      });
    } catch (error) {
      console.error('Code analysis failed:', error);
    }
  };

  const loadExampleCode = () => {
    const examples = {
      performance: `
def fibonacci_slow(n):
    if n <= 1:
        return n
    return fibonacci_slow(n-1) + fibonacci_slow(n-2)

result = fibonacci_slow(30)
print(f"Result: {result}")
`,
      quality: `
def calculate(x,y):
    return x+y

result = calculate(5,3)
`,
      functionality: `
class DataProcessor:
    def process(self, data):
        return data
`,
      security: `
def execute_command(cmd):
    import os
    return os.system(cmd)
`,
      architecture: `
def func1(): pass
def func2(): pass
def func3(): pass
`
    };

    setTargetCode(examples[improvementType] || '');
    setDescription(improvementTypes[improvementType].description);
  };

  return (
    <div className="glass-panel">
      <div className="panel-header">🔧 SELF-MODIFICATION</div>

      {/* Self-Mod Status */}
      <div style={{ marginBottom: '15px', fontSize: '11px', color: '#acc-secondary' }}>
        <div>Proposals: {selfModStatus?.total_proposed || 0}</div>
        <div>Successful: {selfModStatus?.successful_improvements || 0}</div>
        <div>Success Rate: {selfModStatus?.success_rate ? (selfModStatus.success_rate * 100).toFixed(1) : 0}%</div>
      </div>

      {/* Improvement Type Selector */}
      <div style={{ marginBottom: '10px' }}>
        <select
          value={improvementType}
          onChange={(e) => setImprovementType(e.target.value)}
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
          {Object.entries(improvementTypes).map(([key, config]) => (
            <option key={key} value={key}>
              {config.icon} {config.label}
            </option>
          ))}
        </select>
      </div>

      {/* Description Input */}
      <div style={{ marginBottom: '10px' }}>
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Improvement description..."
          style={{
            width: '100%',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            color: '#fff',
            padding: '5px',
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        />
      </div>

      {/* Code Input */}
      <div style={{ marginBottom: '10px' }}>
        <textarea
          value={targetCode}
          onChange={(e) => setTargetCode(e.target.value)}
          placeholder="Paste code to improve..."
          style={{
            width: '100%',
            height: '80px',
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
          onClick={loadExampleCode}
          style={{
            flex: 1,
            background: 'rgba(0,191,255,0.2)',
            border: '1px solid rgba(0,191,255,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '9px',
            cursor: 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          📝 LOAD EXAMPLE
        </button>

        <button
          onClick={analyzeCode}
          disabled={!targetCode.trim()}
          style={{
            flex: 1,
            background: 'rgba(255,215,0,0.2)',
            border: '1px solid rgba(255,215,0,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '9px',
            cursor: targetCode.trim() ? 'pointer' : 'not-allowed',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          🔍 ANALYZE
        </button>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <button
          onClick={startSelfModification}
          disabled={isRunning || !targetCode.trim() || !description.trim()}
          style={{
            width: '100%',
            background: isRunning ? 'rgba(255,165,0,0.3)' : 'rgba(50,205,50,0.2)',
            border: '1px solid rgba(50,205,50,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '10px',
            cursor: (isRunning || !targetCode.trim() || !description.trim()) ? 'not-allowed' : 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          {isRunning ? 'IMPROVING...' : '🚀 START SELF-MODIFICATION'}
        </button>
      </div>

      {/* Last Result Display */}
      {lastResult && (
        <div style={{
          fontSize: '10px',
          color: '#acc-secondary',
          padding: '5px',
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '3px',
          maxHeight: '60px',
          overflowY: 'auto'
        }}>
          <div>Last: {improvementTypes[lastResult.type]?.icon || '🔍'} {lastResult.timestamp}</div>
          {lastResult.result?.success !== undefined && (
            <div style={{ fontSize: '9px', marginTop: '2px' }}>
              Status: {lastResult.result.success ? '✅ Success' : '❌ Failed'}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SelfModControls;


