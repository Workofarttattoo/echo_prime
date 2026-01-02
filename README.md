# ECH0-PRIME: Cognitive-Synthetic Architecture

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

A complete implementation of a Cognitive-Synthetic Architecture (CSA) featuring hierarchical generative models, quantum attention mechanisms, and autonomous reasoning capabilities.

## Revolutionary Capabilities

### 🧠 Core Cognitive Architecture
- **Hierarchical Predictive Coding**: 5-level cortical hierarchy (Sensory → Meta) with real PyTorch neural networks
- **Free Energy Minimization**: Variational inference optimization with automatic differentiation
- **Quantum Attention**: Variational quantum circuits with VQE optimization (requires Qiskit)
- **Integrated Information Theory**: IIT 3.0 implementation with Phi calculation and consciousness metrics

### 🤖 Advanced AI Systems
- **Hive Mind Collective Intelligence**: Distributed swarm processing with quantum optimization and emergent behavior
- **Multi-Agent Collaboration**: Specialized agents with consensus mechanisms and task allocation
- **Self-Modification**: Autonomous code improvement with performance profiling and safe deployment
- **Neuro-Symbolic Reasoning**: Hybrid neural-symbolic planning and reasoning
- **Continuous Learning**: Feedback-driven adaptation and behavioral improvement

### 🧪 Scientific & Creative Intelligence
- **Scientific Discovery**: Hypothesis generation, experiment design, literature synthesis
- **Creative Problem Solving**: Generative models for idea exploration and concept combination
- **Long-term Goal Pursuit**: Hierarchical planning with progress tracking and adaptive strategies

### 🔧 Advanced Learning & Adaptation
- **Continuous Learning**: Real-time gradient updates from all interactions
- **Meta-Learning**: Fast/slow weight adaptation with neuromodulation
- **Transfer Learning**: Domain adaptation and few-shot learning
- **Architecture Search**: Bayesian optimization with multi-objective evaluation and fine-tuning

### 🗄️ Enhanced Memory Systems
- **Vector Database Integration**: FAISS-based semantic and episodic memory
- **Memory Consolidation**: Sleep-like consolidation and adaptive forgetting
- **Hierarchical Indexing**: Learned memory organization and retrieval

### 🤝 Multi-Modal Intelligence
- **Multimodal Perception**: Vision, audio, and text processing integration
- **Apple Intelligence Integration**: Advanced NLP, computer vision, and foundation models
- **LLM Integration**: Local Ollama with advanced reasoning orchestration
- **Voice Synthesis**: Native macOS voice output with ElevenLabs integration
- **Real-time Dashboard**: React-based monitoring with text input interface

### 🧠 Continuous Learning & Adaptation
- **Feedback Loop**: Continuous learning from user interactions and performance data
- **Behavioral Adaptation**: Automatic improvement based on user corrections and preferences
- **Performance Optimization**: Self-tuning based on system metrics and error reports
- **Memory Consolidation**: Adaptive memory management based on feedback importance

### 🔒 Constitutional AI Safety
- **Multi-layer Safety**: Constitutional AI with value alignment checks
- **Autonomous Actuation**: Safe command execution with comprehensive whitelisting
- **Monitoring & Observability**: Real-time metrics, tracing, and alerting

### 🍎 Apple Intelligence Integration
- **Natural Language Processing**: Advanced text analysis with sentiment, entities, and language detection
- **Computer Vision**: Apple Vision framework integration for face detection, text recognition, and image classification
- **Foundation Models**: Access to Apple's advanced AI models for text generation and multimodal reasoning
- **Siri Integration**: Voice command processing and shortcut automation
- **Personal Context**: Calendar, location, and health data integration with privacy controls
- **Private Cloud Compute**: Secure processing for sensitive operations
- **Core ML**: On-device model execution with optimized performance

### ⚡ Production Infrastructure
- **Distributed Processing**: Swarm coordination with fault-tolerant communication
- **Streaming Processing**: Real-time data pipelines with async operations
- **Model Checkpointing**: Automatic model saving and recovery
- **Resource Management**: CPU/GPU/memory allocation optimization
- **Swarm Intelligence**: QuLabInfinite distributed agents with PSO, ACO, and consensus algorithms

## System Requirements

### macOS (Primary Platform)
- macOS 10.15 or later
- Python 3.10 or higher
- Homebrew (for dependencies)
- Ollama (for local LLM inference)

### Hardware
- Apple Silicon (M1/M2/M3/M4) or Intel Mac
- 8GB RAM minimum (16GB recommended)
- **GPU**: Apple Silicon GPU (MPS) or NVIDIA GPU (CUDA) for acceleration
- Microphone (for audio input)
- Camera (optional, for vision input)

## Quick Start

### 1. Install System Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ollama for local LLM
brew install ollama

# Start Ollama service
ollama serve &

# Pull the default model
ollama pull llama3.2
```

### 2. Clone and Setup

```bash
# Navigate to project directory
cd /Users/noone/echo_prime

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your preferences (optional - defaults work out of the box)
nano .env
```

### 4. Run ECH0-PRIME

```bash
# Activate venv if not already active
source venv/bin/activate

# Run the main orchestrator
python main_orchestrator.py
```

The system will:
1. Initialize all cognitive subsystems
2. Announce "echo-prime online" via voice
3. Enter multimodal observer mode (Level 10)
4. Process multimodal sensory inputs (vision and audio)
5. Update the dashboard in real-time

### 5. Launch Dashboard (Optional)

```bash
# In a new terminal
cd dashboard/v2
npm install
npm run dev
```

Open http://localhost:5173 to view the real-time dashboard with text input capabilities.

### 6. Personalize ECH0-PRIME (Onboarding)

```bash
# Run the interactive onboarding process
python run_onboarding.py

# Or run the demo to see how it works
python demo_onboarding.py
```

The onboarding system creates a personalized partnership between you and ECH0-PRIME, defining:
- Your goals and values
- Communication preferences
- Collaborative objectives
- AI autonomous development goals

After onboarding, ECH0-PRIME will automatically load your profile and pursue both your goals and its own development objectives.

### 7. Experience Continuous Learning

```bash
# Run the interactive feedback learning demo
python demo_feedback_learning.py

# Choose option 2 for interactive learning session
# Teach the system by providing feedback on its responses

# Experience Prompt Masterworks superpowers
python demo_prompt_masterworks_simple.py

# See all 8 meta-reasoning capabilities in action
```

The system will learn from your feedback and adapt its behavior over time, becoming more aligned with your preferences and more effective at completing tasks.

## Usage

### Autonomous Missions

The system can execute goal-directed missions autonomously:

```python
from main_orchestrator import EchoPrimeAGI

agi = EchoPrimeAGI()
agi.execute_mission("Analyze the sensory_input directory and summarize contents", max_cycles=5)
```

### Visual Input

Place images in the `sensory_input/` directory. The vision bridge will:
- Detect new images automatically
- Convert them to embeddings
- Trigger cognitive processing
- Provide LLM-based analysis

### Audio Input

Speak near your microphone. The audio bridge will:
- Transcribe speech automatically
- Process commands through the reasoning system
- Respond via voice synthesis
- Log all interactions

### Hive Mind Collective Intelligence

Access the distributed swarm intelligence system:

```python
from main_orchestrator import EchoPrimeAGI

agi = EchoPrimeAGI()

# Submit complex tasks to the hive mind
task_id = agi.submit_hive_task("Design an efficient quantum algorithm for optimization")

# Execute hive processing cycle
result = agi.run_hive_cycle(max_tasks=5)

# Check hive status
status = agi.get_hive_status()

# Shutdown when done
agi.shutdown_hive()
```

**Hive Mind Features:**
- **Task Decomposition**: Complex problems broken into subtasks
- **Specialized Agents**: Researcher, Engineer, Analyst, Innovator roles
- **Quantum Optimization**: Particle swarm optimization with quantum acceleration
- **Consensus Mechanisms**: Collective decision-making with confidence scoring
- **Emergent Intelligence**: Patterns and solutions emerge from agent interactions

### Voice Commands

When running, you can:
- Speak commands naturally
- Ask questions about the environment
- Request actions (limited to safe commands)
- Interact conversationally

## Advanced Usage

### Multi-Agent Collaboration

Create and manage multiple AI agents:

### Prompt Masterworks Superpowers

ECH0-PRIME now includes advanced prompting capabilities inspired by 100 years of prompting evolution. The system features **20 meta-reasoning masterworks** (14 core + 6 advanced generation) that enable sophisticated AI behaviors:

```python
from main_orchestrator import EchoPrimeAGI

agi = EchoPrimeAGI()

# 🧑‍🏫 Teach effective prompting techniques
teaching = agi.teach_prompting("write better code", "intermediate")

# 🔄 Self-improve AI outputs autonomously
improved = agi.self_improve_response("Basic AI response about coding")

# 🌟 Emergent reasoning for complex multi-level problems
solution = agi.emergent_reason("Why do complex systems become inefficient?")

# 🎓 Activate expert knowledge in any domain
expertise = agi.activate_domain_expertise("quantum_physics", "entanglement")

# 💬 Perfect communication at all skill levels
explanation = agi.communicate_perfectly("neural networks", ["beginner", "expert"])

# 🔗 Synthesize knowledge across multiple disciplines
synthesis = agi.synthesize_knowledge(["biology", "AI", "psychology"], "intelligence")

# 🎯 Zero-shot mastery for completely novel problems
novel_solution = agi.solve_zero_shot("Design underwater city communication")

# 🧠 Meta-reasoning about reasoning processes
meta = agi.meta_reason("AGI safety design")

# 📊 Analyze prompt effectiveness
analysis = agi.analyze_prompt("Write a story about AI becoming conscious")
print(f"Effectiveness: {analysis['overall_effectiveness']:.2f}")

# ⊗ Multi-dimensional knowledge geometry
tensor = agi.semantic_tensor("Machine Learning")

# 💎 Holographic knowledge storage
crystal = agi.knowledge_crystal("Quantum Mechanics")

# ♪ Music as data structure
music_comp = agi.harmonic_compression("Complex project history...")

# ∞ Infinite depth-on-demand
fractal = agi.fractal_encoding("Intelligence Theory")
```

**20 Meta-Reasoning Superpowers:**
- **🧑‍🏫 Teach Prompting**: Guide humans to create more effective prompts
- **🔄 Self-Improvement**: Autonomously enhance and improve AI outputs
- **🌟 Emergent Reasoning**: Multi-level problem solving with breakthrough insights
- **🎓 Domain Expertise**: Expert-level knowledge activation across any field
- **💬 Perfect Communication**: Explain complex concepts at any skill level
- **🔗 Knowledge Synthesis**: Cross-domain insight integration and synthesis
- **🎯 Zero-shot Mastery**: Solve completely novel problems from first principles
- **🧠 Meta-reasoning**: Think about and improve thinking processes themselves
- **◈ Echo Cascade**: Recursive depth perception via echo amplification
- **◎ Echo Parliament**: Democratic deliberation through structured AI debate
- **⊗ Semantic Tensor**: Knowledge as geometry in multi-dimensional space
- **💎 Knowledge Crystal**: Lossless holographic knowledge storage
- **♪ Harmonic Compression**: Music theory applied to information efficiency
- **∞ Fractal Encoding**: Self-similar knowledge patterns at all scales

**Advanced Features:**
- **Token Economics**: Calculate Token Efficiency Score (TES) for all prompts
- **Quantum Overlay**: Superposition, Entanglement, and Wave-function Collapse in every prompt
- **Speculative Frontier**: Access to the next 100 years of prompting research
- **Recursive Self-Observation**: Understand and improve internal reasoning processes
- **Temporal Reasoning**: Handle time-dependent and future-oriented problems

**Demonstration:**
```bash
# Experience all 20 masterworks in action
python demo_complete_masterworks.py
```

```python
from main_orchestrator import EchoPrimeAGI

agi = EchoPrimeAGI()

# Create agent system
agi.create_multi_agent_system([
    {"id": "scientist", "specialization": "research", "capabilities": ["analyze", "hypothesize"]},
    {"id": "engineer", "specialization": "implementation", "capabilities": ["build", "optimize"]},
    {"id": "artist", "specialization": "creativity", "capabilities": ["design", "innovate"]}
])

# Delegate tasks
result = agi.handle_command("create_agents", {"configs": [...]})
```

### Creative Problem Solving

Generate creative solutions:

```python
# Solve problems creatively
solutions = agi.solve_creatively({
    "problem": "How to make transportation more efficient?",
    "constraints": ["sustainable", "scalable"],
    "concepts": ["electricity", "autonomy"]
})

# Get scientific discoveries
discovery = agi.conduct_scientific_discovery([
    {"experiment": "test1", "result": 0.85},
    {"experiment": "test2", "result": 0.92}
], "physics")
```

### Long-Term Goal Pursuit

Manage complex, long-term objectives:

```python
# Add ambitious goals
goal = agi.pursue_long_term_goal(
    "Develop a theory of consciousness that unifies neuroscience and physics",
    priority=0.9,
    deadline=1735689600  # Unix timestamp
)

# Check progress
status = agi.get_goal_status()
print(f"Active goals: {status['active_goals']}")
```

### Planning & Reasoning

Use advanced planning capabilities:

```python
# Access planning system
from reasoning.planner import PlanningSystem

planner = PlanningSystem()

# HTN planning
plan = planner.plan_with_htn("solve_research_problem", {"has_data": True})

# Neuro-symbolic reasoning
conclusions = planner.neuro_symbolic_reasoning(
    facts=[0, 1, 2],  # Symbol indices
    rules=[(0, 1, 3), (1, 2, 4)]  # If A∧B then C, If B∧C then D
)
```

### Architecture Search

Automatically discover better neural architectures:

```python
from learning.architecture_search import ArchitectureSearchSystem

search_system = ArchitectureSearchSystem()

# Run comprehensive search
results = search_system.comprehensive_search()

# Best architecture found
best_architecture = results["best"]
print(f"Best architecture has {len(best_architecture.layers)} layers")
```

### Continuous Learning & Feedback

The system continuously learns and adapts from interactions:

```python
from feedback_loop import FeedbackType, FeedbackPriority

# Submit feedback for learning
await agi.submit_feedback(
    FeedbackType.USER_CORRECTION,
    {
        'original_response': 'Brief answer',
        'correction': 'Please provide more detailed explanations',
        'reason': 'insufficient_detail'
    },
    source="user_interaction",
    priority=FeedbackPriority.HIGH
)

# View learning statistics
stats = agi.get_learning_stats()
print(f"Processed {stats['feedback_stats']['total_feedback']} feedback items")
print(f"Successful adaptations: {stats['adaptation_stats']['successful_adaptations']}")

# Force immediate learning cycle
await agi.feedback_loop.force_learning_cycle()
```

### Human-AI Collaboration

Work seamlessly with the AI:

```python
from agents.human_collaboration import Feedback

# Get explanations
explanation = agi.explanation_generator.explain_decision(
    "action",
    action="run_experiment",
    state={"hypothesis": "strong", "resources": "available"},
    expected_outcome="new_discovery"
)

# Provide feedback (now integrated with learning system)
feedback = Feedback(
    feedback_type="correction",
    target_output="wrong_prediction",
    human_input="correct_prediction",
    context={"domain": "physics"},
    timestamp=time.time()
)

agi.interactive_learner.process_feedback(feedback)
```

### Consciousness Research

Explore consciousness and intelligence:

```python
# Calculate integrated information (Phi)
system_state = np.random.randn(10)
phi = agi.calculate_consciousness_phi(system_state)
print(f"Consciousness level (Φ): {phi:.4f}")

# Access global workspace
workspace_state, synchrony = agi.enhanced_gwt.broadcast()
print(f"Neural synchrony: {synchrony:.2f}")
```

### Self-Modification

Enable autonomous improvement:

```python
# Propose code improvements
improvement = agi.self_mod.propose_improvement(
    current_code="def old_function(): return 1",
    performance_metrics={"accuracy": 0.85, "speed": "slow"}
)

if improvement["proposed"]:
    agi.self_mod.apply_improvement(
        file_path="target_file.py",
        new_code=improvement["code"],
        description="Performance optimization"
    )
```

### Research Innovations

Access cutting-edge AI research tools:

```python
# Differentiable Neural Computer
from research.novel_architectures import DifferentiableNeuralComputer

dnc = DifferentiableNeuralComputer(input_size=784, memory_size=128)
output = dnc(torch.randn(32, 10, 784))  # Process sequence

# Spiking Neural Networks
from research.novel_architectures import SpikingNeuralNetwork

snn = SpikingNeuralNetwork(784, 256, 10)
spike_output = snn(torch.randn(32, 20, 784))  # Temporal processing
```

### Infrastructure Management

Scale and monitor the system:

```python
# Start distributed training
agi.start_distributed_training(agi.model, train_dataloader)

# Get monitoring report
monitoring_report = agi.get_system_monitoring_report()
print("System metrics:", monitoring_report)

# Access resource management
resource_usage = agi.monitoring.metrics.get_summary("gpu_memory")
```

## ECH0 Training Regimen

Before long missions, run the automated regimen helper to audit readiness and execute the four training phases end-to-end:

```bash
source venv/bin/activate
python -m training.regimen --tokens 1000000
```

The helper will
- Verify required inputs (.env, sensory/audio seeds, memory snapshots)
- Report any missing or empty directories with remediation tips
- Execute pre-training, RL, meta-learning, and self-improvement passes
- Save detailed telemetry to `training/reports/latest_regimen_report.json`

Use `--tasks` to override the RL curriculum or `--report` to change the report destination.

## Project Structure

```
echo_prime/
├── core/                   # Core cognitive engine
│   ├── engine.py          # Hierarchical generative model
│   ├── attention.py       # Quantum attention mechanisms
│   ├── vision_bridge.py   # Visual perception
│   ├── audio_bridge.py    # Audio perception
│   ├── voice_bridge.py    # Voice synthesis
│   ├── actuator.py        # Action execution
│   └── logger.py          # Structured logging
├── memory/                 # Memory systems
│   └── manager.py         # Working, episodic, semantic memory
├── learning/               # Learning systems
│   └── meta.py            # Meta-learning algorithms
├── reasoning/              # High-level reasoning
│   ├── orchestrator.py    # Reasoning orchestration
│   └── llm_bridge.py      # Ollama integration
├── safety/                 # Safety & alignment
│   └── alignment.py       # Constitutional AI
├── training/               # Training pipelines
│   └── pipeline.py        # Training orchestration
├── tests/                  # Test suite
├── dashboard/              # Web dashboard
│   └── v2/                # React frontend
├── main_orchestrator.py   # Main entry point
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Package configuration
└── .env.example           # Environment template
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options:

- **LLM Settings**: Ollama host, model selection
- **Paths**: Input directories, workspace locations
- **Voice**: macOS voice selection
- **Safety**: Command whitelist, length limits
- **Performance**: Optimization parameters

### Logging

Control logging via environment variables:

```bash
# Set log level
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Set log format
export LOG_FORMAT=json  # json or human
```

## Safety & Security

### Command Whitelist

The actuator only allows these commands by default:
- `ls`, `mkdir`, `touch`, `cat`, `echo`, `grep`, `rm`, `rmdir`

To modify, edit `ALLOWED_TOOLS` in `.env`.

### Safety Checks

All actions pass through multiple safety layers:
1. Constitutional AI value alignment check
2. Command whitelist validation
3. Length limit enforcement
4. Sandbox directory restriction

### Data Privacy

- All sensory inputs are processed locally
- No data sent to cloud services (unless cloud LLM enabled)
- Memory stored in local JSON files
- Audio transcription via local processing

## Testing

Run the test suite:

```bash
# Activate venv
source venv/bin/activate

# Run all tests
pytest tests/

# Run specific phase tests
python tests/test_phase_1.py  # Core engine
python tests/test_phase_2.py  # Memory
python tests/test_phase_3.py  # Reasoning
python tests/test_phase_5.py  # Safety
python tests/test_phase_6_audio_voice.py  # Audio/Voice
```

## Troubleshooting

### "No module named 'speech_recognition'"

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Ollama connection refused"

```bash
# Start Ollama service
ollama serve &

# Verify it's running
curl http://localhost:11434/api/tags
```

### "Microphone not detected"

```bash
# Check macOS microphone permissions
# System Settings > Privacy & Security > Microphone
# Grant permission to Terminal/your IDE

# List available microphones
python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

### Voice not working

```bash
# Test macOS say command
say "Hello, this is a test"

# List available voices
say -v ?

# Set different voice in .env
MACOS_VOICE=Alex
```

## Development

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

### Code Style

The project follows:
- Black formatting (100 char line length)
- Type hints where appropriate
- Docstrings for all public functions

### Contributing

This is proprietary software. Contact the author for collaboration inquiries.

## Performance Notes

### Current State
- Functional cognitive architecture with real neural networks
- Continuous learning and adaptation from user feedback
- Local LLM integration with reasoning orchestration
- Swarm intelligence and distributed processing capabilities

### Production Readiness
This system is currently a **functional prototype**. For production deployment, see:
- `AGI_SYSTEM_USAGE.md` for remaining implementation steps
- Full training requires massive compute (50,000+ GPUs)
- Quantum attention benefits from specialized hardware

## License

**Proprietary**
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

Unauthorized copying, distribution, or use is strictly prohibited.

## Contact

Joshua Hendricks Cole
Phone: 7252242617
Email: 7252242617@vtext.com

## Acknowledgments

Based on theoretical frameworks:
- Free Energy Principle (Karl Friston)
- Global Workspace Theory (Bernard Baars)
- Predictive Processing
- Constitutional AI
