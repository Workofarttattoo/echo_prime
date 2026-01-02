import React, { useState, useEffect } from 'react';
import MetricCard from './components/MetricCard';
import Visualizer from './components/Visualizer';
import SensoryWindow from './components/SensoryWindow';
import MemoryWindow from './components/MemoryWindow';
import BenchmarkWindow from './components/BenchmarkWindow';
import TextInput from './components/TextInput';
import './index.css';

function App() {
  const defaultHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
  const defaultWs = import.meta.env.VITE_WS_URL || `ws://${defaultHost}:8000/ws`;
  const defaultApi = import.meta.env.VITE_API_URL || `http://${defaultHost}:8000`;

  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [benchmarks, setBenchmarks] = useState({});
  const [wsUrl, setWsUrl] = useState(defaultWs);
  const [apiBaseUrl, setApiBaseUrl] = useState(defaultApi);

  useEffect(() => {
    const fetchBenchmarks = async () => {
      try {
        const res = await fetch(`${window.location.origin}/data/benchmarks.json`);
        if (res.ok) {
          const json = await res.json();
          setBenchmarks(json);
        }
      } catch (e) { }
    };

    fetchBenchmarks();
    const interval = setInterval(fetchBenchmarks, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchWsConfig = async () => {
      try {
        const res = await fetch('/data/api_port.json', { cache: 'no-store' });
        if (res.ok) {
          const cfg = await res.json();
          if (cfg.ws_url) setWsUrl(cfg.ws_url);
          if (cfg.port) setApiBaseUrl(`http://${defaultHost}:${cfg.port}`);
        }
      } catch (e) { }
    };
    fetchWsConfig();
  }, [defaultHost]);

  useEffect(() => {
    let socket;
    let reconnectTimeout;

    const connect = () => {
      console.log('Attempting WebSocket connection...');
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        console.log('WebSocket connected');
        setError(null);
      };

      socket.onmessage = (event) => {
        const json = JSON.parse(event.data);
        setData(json);
        setError(null);

        // Maintain local history of thoughts
        if (json.reasoning?.insight) {
          setHistory(prev => {
            const last = prev[prev.length - 1]?.text;
            if (last !== json.reasoning.insight) {
              const newEntry = {
                text: json.reasoning.insight,
                time: new Date().toLocaleTimeString()
              };
              return [...prev.slice(-999), newEntry];
            }
            return prev;
          });
        }
      };

      socket.onerror = (err) => {
        console.error('WebSocket Error:', err);
      };

      socket.onclose = () => {
        console.log('WebSocket Closed. Reconnecting in 2s...');
        setError('Connection Lost. Reconnecting...');
        reconnectTimeout = setTimeout(connect, 2000);
      };
    };

    if (wsUrl) connect();

    return () => {
      if (socket) socket.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [wsUrl]);

  // Calculate System Strain for Visualizer
  // Strain = Free Energy / Threshold (roughly)
  const fe = data?.engine?.free_energy || 0;
  const strain = Math.min(Math.max(fe / 100, 0), 1.0);

  // Mute Logic
  const toggleMute = async () => {
    const currentState = data?.engine?.voice_enabled ?? true;
    try {
      await fetch(`${apiBaseUrl}/mute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mute: currentState })
      });
      // Optimistic update
      setData(prev => ({
        ...prev,
        engine: { ...prev.engine, voice_enabled: !currentState }
      }));
    } catch (e) {
      console.error("Mute failed:", e);
    }
  };

  const startListening = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Voice input not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);

    recognition.onresult = async (event) => {
      const text = event.results[0][0].transcript;
      try {
        await fetch(`${apiBaseUrl}/speech`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
      } catch (e) {
        console.error("Speech send failed:", e);
      }
    };

    recognition.start();
  };


  if (error) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', color: '#ff0055', background: '#000' }}>
        <div style={{ textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
          <h1>SYSTEM OFFLINE</h1>
          <p>{">"} LINK_FAILURE: {error}</p>
          <div className="resonance-ring" style={{ animationDuration: '0.5s', borderColor: '#ff0055' }} />
        </div>
      </div>
    );
  }

  const isMuted = data?.engine?.voice_enabled === false;

  return (
    <div className="dashboard-container">

      {/* BACKGROUND VISUALIZER */}
      <Visualizer strain={strain} coherence={data?.attention?.coherence || 1.0} />

      {/* LEFT COLUMN: SENSORY & METRICS */}
      <div className="dashboard-column">
        <div className="glass-panel panel-header-container">
          <div className="title-section">
            <h1>ECH0-PRIME</h1>
            <div className="status-tag">
              STATUS: {data?.engine?.surprise.includes("SURPRISE") ? "ADAPTING" : "OPTIMAL"}
            </div>
          </div>
          <div className="button-group">
            <button
              onClick={startListening}
              style={{
                background: isListening ? 'rgba(0, 242, 255, 0.4)' : 'rgba(255, 255, 255, 0.05)',
                border: `1px solid ${isListening ? '#00f2ff' : 'rgba(255,255,255,0.2)'}`,
                color: '#fff',
                padding: '5px 10px',
                fontSize: '10px',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
                animation: isListening ? 'pulse 1s infinite' : 'none'
              }}
            >
              {isListening ? "LISTENING..." : "🎙️ TALK"}
            </button>
            <button
              onClick={toggleMute}
              style={{
                background: isMuted ? 'rgba(255, 0, 85, 0.2)' : 'rgba(0, 255, 136, 0.2)',
                border: `1px solid ${isMuted ? '#ff0055' : '#00ff88'}`,
                color: '#fff',
                padding: '5px 10px',
                fontSize: '10px',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)'
              }}
            >
              {isMuted ? "UNMUTE" : "MUTE"}
            </button>
          </div>
        </div>

        <SensoryWindow
          activeVisual={data?.sensory?.active_visual}
          isListening={data?.sensory?.audio_input_detected}
        />

        <TextInput
          apiBaseUrl={apiBaseUrl}
          onSend={(text) => {
            console.log('Text sent:', text);
            // Optionally update history or trigger UI updates
          }}
        />

        <BenchmarkWindow benchmarks={benchmarks} />

        <div className="glass-panel">
          <MetricCard label="FREE ENERGY" value={data?.engine?.free_energy?.toFixed(2) || "0.00"} />
          <MetricCard label="COHERENCE" value={data?.attention?.coherence?.toFixed(2) || "1.00"} />
          <MetricCard label="PHI LEVEL" value={data?.engine?.phi?.toFixed(2) || "0.00"} />
          <MetricCard label="EPISODIC MEM" value={data?.memory?.episodic_count || 0} />
        </div>
      </div>

      {/* CENTER COLUMN: ACTIVE GOAL (Float) */}
      <div className="dashboard-column center-column">
        {data?.engine?.mission_goal && (
          <div className="glass-panel directive-panel">
            <div className="directive-label">CURRENT DIRECTIVE</div>
            <div className="directive-text">
              {data.engine.mission_goal}
            </div>
            {data.engine.mission_complete && <div className="achieved-tag">[ACHIEVED]</div>}
          </div>
        )}
      </div>

      {/* RIGHT COLUMN: COGNITION */}
      <div className="dashboard-column">
        <MemoryWindow
          thoughts={history}
          memories={data?.memory?.recent_notes || []}
          semanticCount={data?.memory?.semantic_concepts || 0}
        />

        <div className="glass-panel" style={{ height: '200px', overflowY: 'auto' }}>
          <div className="panel-header">ACTUATOR LOG</div>
          {data?.reasoning?.actions.map((act, i) => (
            <div key={i} style={{ fontSize: '11px', marginBottom: '5px', color: '#acc-secondary' }}>
              {">"} {act}
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

export default App;

