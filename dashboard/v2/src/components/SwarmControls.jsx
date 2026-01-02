import React, { useState, useEffect } from 'react';

const SwarmControls = ({ onSwarmCommand, swarmStatus }) => {
  const [algorithm, setAlgorithm] = useState('pso');
  const [problemDim, setProblemDim] = useState(10);
  const [iterations, setIterations] = useState(50);
  const [isRunning, setIsRunning] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const algorithms = {
    pso: {
      name: 'Particle Swarm Optimization',
      description: 'Distributed optimization using swarm behavior',
      icon: '🐝'
    },
    aco: {
      name: 'Ant Colony Optimization',
      description: 'Pheromone-based path finding',
      icon: '🐜'
    },
    consensus: {
      name: 'Swarm Consensus',
      description: 'Collective decision making',
      icon: '🤝'
    }
  };

  const startSwarmOptimization = async () => {
    setIsRunning(true);
    try {
      const result = await onSwarmCommand('optimize', {
        algorithm,
        problem_dimension: problemDim,
        iterations
      });

      setLastResult({
        algorithm,
        result,
        timestamp: new Date().toLocaleTimeString()
      });
    } catch (error) {
      console.error('Swarm optimization failed:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const createSwarmAgent = async () => {
    try {
      const agentId = await onSwarmCommand('create_agent', {
        specialization: 'dashboard_created',
        capabilities: ['computation', 'communication', 'optimization']
      });
      console.log('Created swarm agent:', agentId);
    } catch (error) {
      console.error('Agent creation failed:', error);
    }
  };

  return (
    <div className="glass-panel">
      <div className="panel-header">🐝 SWARM INTELLIGENCE</div>

      {/* Swarm Status */}
      <div style={{ marginBottom: '15px', fontSize: '11px', color: '#acc-secondary' }}>
        <div>Agents: {swarmStatus?.total_agents || 0}</div>
        <div>Active: {swarmStatus?.active_agents || 0}</div>
        <div>Best Fitness: {swarmStatus?.global_best_fitness?.toFixed(4) || 'N/A'}</div>
      </div>

      {/* Algorithm Selection */}
      <div style={{ marginBottom: '10px' }}>
        <select
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value)}
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
          {Object.entries(algorithms).map(([key, config]) => (
            <option key={key} value={key}>
              {config.icon} {config.name}
            </option>
          ))}
        </select>
      </div>

      {/* Algorithm Description */}
      <div style={{
        fontSize: '10px',
        color: '#acc-secondary',
        marginBottom: '10px',
        minHeight: '20px'
      }}>
        {algorithms[algorithm].description}
      </div>

      {/* Parameters */}
      <div style={{ marginBottom: '10px' }}>
        <div style={{ display: 'flex', gap: '5px', marginBottom: '5px' }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: '9px', color: '#acc-secondary' }}>Dimensions:</label>
            <input
              type="number"
              value={problemDim}
              onChange={(e) => setProblemDim(parseInt(e.target.value))}
              min="2"
              max="100"
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.2)',
                color: '#fff',
                padding: '3px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                borderRadius: '3px'
              }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: '9px', color: '#acc-secondary' }}>Iterations:</label>
            <input
              type="number"
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value))}
              min="10"
              max="1000"
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.2)',
                color: '#fff',
                padding: '3px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                borderRadius: '3px'
              }}
            />
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
        <button
          onClick={startSwarmOptimization}
          disabled={isRunning}
          style={{
            flex: 1,
            background: isRunning ? 'rgba(255,165,0,0.3)' : 'rgba(138,43,226,0.2)',
            border: '1px solid rgba(138,43,226,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '10px',
            cursor: isRunning ? 'not-allowed' : 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          {isRunning ? 'OPTIMIZING...' : '🚀 START SWARM'}
        </button>

        <button
          onClick={createSwarmAgent}
          style={{
            flex: 1,
            background: 'rgba(255,20,147,0.2)',
            border: '1px solid rgba(255,20,147,0.5)',
            color: '#fff',
            padding: '5px',
            fontSize: '10px',
            cursor: 'pointer',
            fontFamily: 'var(--font-mono)',
            borderRadius: '3px'
          }}
        >
          ➕ ADD AGENT
        </button>
      </div>

      {/* Last Result Display */}
      {lastResult && (
        <div style={{
          fontSize: '10px',
          color: '#acc-secondary',
          padding: '5px',
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '3px'
        }}>
          <div>Last: {algorithms[lastResult.algorithm].icon} {lastResult.timestamp}</div>
          <div style={{ fontSize: '9px', marginTop: '2px' }}>
            Best Fitness: {lastResult.result?.best_fitness?.toFixed(4) || 'N/A'}
          </div>
        </div>
      )}
    </div>
  );
};

export default SwarmControls;


