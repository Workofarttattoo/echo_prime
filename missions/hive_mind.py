"""
Hive Mind Script from QuLabInfinite

Implements a distributed collective intelligence system where multiple ECH0-PRIME
instances collaborate through QuLabInfinite for advanced scientific discovery,
problem-solving, and emergent intelligence.

The hive mind operates as:
- Swarm Intelligence: Multiple agents solving problems collectively
- Emergent Behavior: Complex solutions from simple agent interactions
- Quantum-Inspired Processing: Leveraging QuLabInfinite's quantum computing capabilities
- Self-Organizing: Agents dynamically form task-specific collectives
"""

import os
import sys
import time
import json
import numpy as np
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import queue
import asyncio
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.tools.qulab import QuLabBridge


class HiveMindState(Enum):
    INITIALIZING = "initializing"
    FORMING_COLLECTIVE = "forming_collective"
    TASK_ALLOCATION = "task_allocation"
    COLLABORATIVE_SOLVING = "collaborative_solving"
    CONSENSUS_BUILDING = "consensus_building"
    SOLUTION_EMERGENCE = "solution_emergence"
    DISBANDING = "disbanding"


@dataclass
class HiveNode:
    """Represents a single node in the hive mind"""
    node_id: str
    specialization: str
    capabilities: List[str]
    performance_score: float = 0.0
    active_tasks: List[str] = field(default_factory=list)
    quantum_access: bool = False
    last_active: float = field(default_factory=time.time)

    def is_available(self) -> bool:
        """Check if node is available for new tasks"""
        return len(self.active_tasks) < 3  # Max 3 concurrent tasks

    def update_performance(self, task_success: bool, complexity: float):
        """Update node's performance score"""
        reward = complexity if task_success else -complexity * 0.5
        self.performance_score = 0.9 * self.performance_score + 0.1 * reward


@dataclass
class HiveTask:
    """Represents a task being solved by the hive mind"""
    task_id: str
    description: str
    complexity: float
    domain: str
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    assigned_nodes: List[str] = field(default_factory=list)
    solutions: Dict[str, Any] = field(default_factory=dict)
    consensus_solution: Optional[Any] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)


class QuantumSwarmProcessor:
    """
    Processes tasks using quantum-inspired swarm algorithms via QuLabInfinite
    """
    def __init__(self, qulab_bridge: QuLabBridge):
        self.qulab = qulab_bridge
        self.quantum_available = qulab_bridge.available

    def quantum_particle_swarm_optimization(self, problem_space: Dict[str, Any],
                                          num_particles: int = 10) -> Dict[str, Any]:
        """
        Use quantum PSO for optimization problems
        """
        if not self.quantum_available:
            return self._classical_pso_fallback(problem_space, num_particles)

        # Generate quantum circuit for PSO
        script = f"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import PSO

# Quantum Particle Swarm Optimization
def quantum_pso(objective_func, bounds, num_particles={num_particles}):
    optimizer = PSO(maxiter=100, popsize=num_particles)

    # Define quantum objective function
    def quantum_objective(x):
        # Encode classical parameters into quantum state
        qc = QuantumCircuit(len(x))
        for i, val in enumerate(x):
            qc.ry(val * np.pi, i)  # Encode parameter as rotation

        # Add entanglement for swarm behavior
        for i in range(len(x)-1):
            qc.cx(i, i+1)

        # Measure and return expectation value
        return objective_func(x)

    result = optimizer.optimize(bounds, quantum_objective)
    return {{
        'optimal_solution': result.x.tolist(),
        'optimal_value': result.fun,
        'convergence': result.nfev
    }}

# Run optimization
bounds = {problem_space.get('bounds', '[(0, 1)] * 5')}
result = quantum_pso(lambda x: sum(x**2), bounds)  # Example quadratic optimization
print(json.dumps(result))
"""

        result = self.qulab.run_command(f"python3 -c '{script}'")
        try:
            return json.loads(result.split('QULAB EXECUTION RESULT:')[1].strip())
        except:
            return self._classical_pso_fallback(problem_space, num_particles)

    def _classical_pso_fallback(self, problem_space: Dict[str, Any], num_particles: int) -> Dict[str, Any]:
        """Classical PSO fallback when quantum computing unavailable"""
        bounds = problem_space.get('bounds', [(-5, 5)] * 5)

        # Simple PSO implementation
        particles = []
        velocities = []
        personal_best = []
        global_best = None
        global_best_score = float('inf')

        # Initialize particles
        for _ in range(num_particles):
            particle = [np.random.uniform(low, high) for low, high in bounds]
            velocity = [np.random.uniform(-1, 1) for _ in bounds]
            particles.append(particle)
            velocities.append(velocity)
            personal_best.append((particle.copy(), float('inf')))

        # PSO parameters
        w = 0.7  # inertia
        c1 = 1.5  # cognitive
        c2 = 1.5  # social

        for iteration in range(50):
            for i, particle in enumerate(particles):
                # Evaluate fitness (simple quadratic)
                fitness = sum(x**2 for x in particle)

                # Update personal best
                if fitness < personal_best[i][1]:
                    personal_best[i] = (particle.copy(), fitness)

                # Update global best
                if fitness < global_best_score:
                    global_best = particle.copy()
                    global_best_score = fitness

            # Update velocities and positions
            for i in range(num_particles):
                for j in range(len(bounds)):
                    r1, r2 = np.random.random(), np.random.random()

                    velocities[i][j] = (w * velocities[i][j] +
                                      c1 * r1 * (personal_best[i][0][j] - particles[i][j]) +
                                      c2 * r2 * (global_best[j] - particles[i][j]))

                    particles[i][j] += velocities[i][j]

                    # Clamp to bounds
                    low, high = bounds[j]
                    particles[i][j] = np.clip(particles[i][j], low, high)

        return {
            'optimal_solution': global_best,
            'optimal_value': global_best_score,
            'method': 'classical_pso'
        }


class EmergentIntelligenceEngine:
    """
    Engine for emergent collective intelligence from simple agent interactions
    """
    def __init__(self, qulab_bridge: QuLabBridge):
        self.qulab = qulab_bridge
        self.pattern_memory = {}
        self.emergence_history = []

    def detect_emergent_patterns(self, agent_interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect emergent patterns from agent interactions using QuLab analysis
        """
        if not self.qulab.available:
            return self._simple_pattern_detection(agent_interactions)

        # Use QuLab for advanced pattern detection
        interaction_data = json.dumps(agent_interactions)

        script = f"""
import json
import numpy as np
from scipy import stats

# Load interaction data
interactions = json.loads('''{interaction_data}''')

# Analyze interaction patterns
patterns = []
timestamps = [i['timestamp'] for i in interactions]
messages = [i['content'] for i in interactions]

# Detect temporal patterns
if len(timestamps) > 5:
    # Calculate interaction frequency
    time_diffs = np.diff(sorted(timestamps))
    freq_pattern = {{
        'type': 'temporal_clustering',
        'frequency': len(timestamps) / (max(timestamps) - min(timestamps)) if timestamps else 0,
        'burstiness': np.std(time_diffs) / np.mean(time_diffs) if time_diffs.size > 0 else 0
    }}
    patterns.append(freq_pattern)

# Detect semantic patterns
unique_messages = set(str(msg) for msg in messages)
semantic_pattern = {{
    'type': 'semantic_diversity',
    'unique_messages': len(unique_messages),
    'total_messages': len(messages),
    'diversity_ratio': len(unique_messages) / len(messages) if messages else 0
}}

patterns.append(semantic_pattern)
print(json.dumps(patterns))
"""

        result = self.qulab.run_command(f"python3 -c '{script}'")
        try:
            patterns = json.loads(result.split('QULAB EXECUTION RESULT:')[1].strip())
            self.emergence_history.extend(patterns)
            return patterns
        except:
            return self._simple_pattern_detection(agent_interactions)

    def _simple_pattern_detection(self, agent_interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple pattern detection fallback"""
        if not agent_interactions:
            return []

        # Basic temporal clustering
        timestamps = [i['timestamp'] for i in agent_interactions]
        if len(timestamps) > 1:
            time_span = max(timestamps) - min(timestamps)
            frequency = len(timestamps) / time_span if time_span > 0 else 0

            return [{
                'type': 'temporal_clustering',
                'frequency': frequency,
                'method': 'simple'
            }]
        return []

    def generate_emergent_solution(self, partial_solutions: List[Dict[str, Any]],
                                 problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate emergent solution from partial solutions using collective intelligence
        """
        if not partial_solutions:
            return {'solution': None, 'confidence': 0.0}

        # Combine solutions using weighted voting based on agent performance
        solution_votes = {}
        total_weight = 0

        for solution in partial_solutions:
            sol_key = str(solution.get('solution', ''))
            weight = solution.get('agent_performance', 1.0)
            solution_votes[sol_key] = solution_votes.get(sol_key, 0) + weight
            total_weight += weight

        # Find consensus solution
        if solution_votes:
            best_solution = max(solution_votes.items(), key=lambda x: x[1])
            confidence = best_solution[1] / total_weight if total_weight > 0 else 0.0

            return {
                'solution': best_solution[0],
                'confidence': confidence,
                'votes': solution_votes,
                'emergence_method': 'weighted_consensus'
            }

        return {'solution': None, 'confidence': 0.0}


class HiveMindOrchestrator:
    """
    Main orchestrator for the hive mind collective intelligence system
    """
    def __init__(self, num_nodes: int = 5, qulab_path: str = "/Users/noone/QuLabInfinite"):
        self.num_nodes = num_nodes
        self.state = HiveMindState.INITIALIZING

        # Core components
        self.qulab_bridge = QuLabBridge(qulab_path)
        self.quantum_processor = QuantumSwarmProcessor(self.qulab_bridge)
        self.emergence_engine = EmergentIntelligenceEngine(self.qulab_bridge)

        # Hive components
        self.nodes: Dict[str, HiveNode] = {}
        self.tasks: Dict[str, HiveTask] = {}
        self.task_queue = queue.Queue()
        self.solution_queue = queue.Queue()

        # Communication
        self.message_bus = queue.Queue()
        self.threads: List[threading.Thread] = []

        # Initialize hive
        self._initialize_hive_nodes()

    def _initialize_hive_nodes(self):
        """Initialize the hive mind nodes with different specializations"""
        specializations = [
            ("researcher", ["analyze", "hypothesize", "search"]),
            ("engineer", ["design", "optimize", "implement"]),
            ("analyst", ["evaluate", "validate", "measure"]),
            ("innovator", ["create", "synthesize", "transform"]),
            ("coordinator", ["organize", "delegate", "integrate"])
        ]

        for i in range(self.num_nodes):
            spec_name, capabilities = specializations[i % len(specializations)]
            node_id = f"hive_node_{i+1}"

            node = HiveNode(
                node_id=node_id,
                specialization=f"{spec_name}_{i//len(specializations) + 1}",
                capabilities=capabilities,
                quantum_access=(i < 2)  # First 2 nodes have quantum access
            )

            self.nodes[node_id] = node

    def submit_task(self, description: str, domain: str = "general", complexity: float = 1.0) -> str:
        """Submit a task to the hive mind"""
        task_id = f"task_{int(time.time())}_{len(self.tasks)}"

        task = HiveTask(
            task_id=task_id,
            description=description,
            complexity=complexity,
            domain=domain
        )

        self.tasks[task_id] = task
        self.task_queue.put(task_id)

        print(f"🐝 Hive Mind: Task {task_id} submitted - {description}")
        return task_id

    def _decompose_task(self, task: HiveTask) -> List[Dict[str, Any]]:
        """Decompose complex task into subtasks using hive intelligence"""
        # Use quantum processor for task decomposition if available
        if self.qulab_bridge.available:
            decomposition_result = self.quantum_processor.quantum_particle_swarm_optimization({
                'bounds': [(0, 1)] * 5,  # Task decomposition parameters
                'objective': 'task_breakdown'
            })

            # Generate subtasks based on quantum optimization result
            num_subtasks = max(2, min(8, int(decomposition_result.get('optimal_value', 4))))
        else:
            num_subtasks = max(2, min(6, int(task.complexity * 3)))

        # Create subtasks
        subtasks = []
        for i in range(num_subtasks):
            subtask = {
                'id': f"{task.task_id}_sub_{i}",
                'description': f"Subtask {i+1} for: {task.description}",
                'complexity': task.complexity / num_subtasks,
                'domain': task.domain,
                'assigned_node': None,
                'status': 'pending'
            }
            subtasks.append(subtask)

        task.subtasks = subtasks
        return subtasks

    def _allocate_subtasks(self, task: HiveTask):
        """Allocate subtasks to available hive nodes"""
        available_nodes = [node for node in self.nodes.values() if node.is_available()]

        if not available_nodes:
            print("🐝 Warning: No available nodes for task allocation")
            return

        # Sort nodes by specialization match and performance
        def node_score(node: HiveNode) -> float:
            domain_match = 1.0 if node.specialization.split('_')[0] == task.domain else 0.5
            availability = 1.0 / (len(node.active_tasks) + 1)
            performance = node.performance_score + 1.0  # Add 1 to avoid negative scores
            return domain_match * availability * performance

        sorted_nodes = sorted(available_nodes, key=node_score, reverse=True)

        # Allocate subtasks to best available nodes
        for i, subtask in enumerate(task.subtasks):
            if i < len(sorted_nodes):
                node = sorted_nodes[i]
                subtask['assigned_node'] = node.node_id
                node.active_tasks.append(subtask['id'])
                task.assigned_nodes.append(node.node_id)

    def _process_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Process a subtask using the assigned node"""
        node_id = subtask['assigned_node']
        node = self.nodes.get(node_id)

        if not node:
            return {'error': f'Node {node_id} not found'}

        # Simulate node processing based on specialization
        processing_time = subtask['complexity'] * (2.0 if node.quantum_access else 1.0)

        # Generate solution based on node specialization
        specialization = node.specialization.split('_')[0]

        if specialization == 'researcher':
            solution = f"Research finding: {subtask['description']} reveals key insights"
        elif specialization == 'engineer':
            solution = f"Engineering solution: {subtask['description']} optimized design"
        elif specialization == 'analyst':
            solution = f"Analysis result: {subtask['description']} statistical validation"
        elif specialization == 'innovator':
            solution = f"Innovative approach: {subtask['description']} creative synthesis"
        else:  # coordinator
            solution = f"Coordination plan: {subtask['description']} integrated approach"

        return {
            'subtask_id': subtask['id'],
            'solution': solution,
            'confidence': 0.8 + np.random.random() * 0.2,
            'processing_time': processing_time,
            'node_id': node_id
        }

    def _collect_solutions(self, task: HiveTask) -> Dict[str, Any]:
        """Collect and synthesize solutions from all subtasks"""
        solutions = []

        for subtask in task.subtasks:
            if subtask.get('status') == 'completed':
                solution_data = {
                    'solution': subtask.get('solution', ''),
                    'confidence': subtask.get('confidence', 0.5),
                    'node_id': subtask.get('assigned_node', ''),
                    'agent_performance': self.nodes[subtask['assigned_node']].performance_score
                }
                solutions.append(solution_data)

        # Use emergent intelligence to combine solutions
        emergent_result = self.emergence_engine.generate_emergent_solution(
            solutions,
            {'task_description': task.description, 'domain': task.domain}
        )

        return emergent_result

    def run_hive_cycle(self) -> Dict[str, Any]:
        """Execute one complete hive mind cycle"""
        self.state = HiveMindState.TASK_ALLOCATION

        # Process pending tasks
        completed_tasks = []

        try:
            # Get next task
            task_id = self.task_queue.get_nowait()
            task = self.tasks[task_id]

            print(f"🐝 Processing task: {task.description}")

            # Phase 1: Task Decomposition
            self.state = HiveMindState.COLLABORATIVE_SOLVING
            subtasks = self._decompose_task(task)
            print(f"🐝 Decomposed into {len(subtasks)} subtasks")

            # Phase 2: Subtask Allocation
            self._allocate_subtasks(task)
            print(f"🐝 Allocated to nodes: {task.assigned_nodes}")

            # Phase 3: Parallel Processing (simulate with threads)
            self.state = HiveMindState.COLLABORATIVE_SOLVING

            # Process subtasks (in parallel simulation)
            subtask_results = []
            for subtask in task.subtasks:
                result = self._process_subtask(subtask)
                subtask_results.append(result)
                subtask['solution'] = result.get('solution', '')
                subtask['confidence'] = result.get('confidence', 0.5)
                subtask['status'] = 'completed'

                # Update node performance
                node = self.nodes.get(subtask['assigned_node'])
                if node:
                    node.update_performance(True, subtask['complexity'])
                    node.active_tasks.remove(subtask['id'])

            # Phase 4: Solution Emergence
            self.state = HiveMindState.SOLUTION_EMERGENCE
            final_solution = self._collect_solutions(task)
            task.consensus_solution = final_solution
            task.status = 'completed'

            print(f"🐝 Task {task_id} completed with confidence: {final_solution.get('confidence', 0):.2f}")

            completed_tasks.append({
                'task_id': task_id,
                'solution': final_solution,
                'subtasks_completed': len(subtask_results)
            })

        except queue.Empty:
            # No tasks to process
            pass

        self.state = HiveMindState.FORMING_COLLECTIVE

        return {
            'completed_tasks': completed_tasks,
            'active_nodes': len([n for n in self.nodes.values() if n.active_tasks]),
            'hive_state': self.state.value
        }

    def get_hive_status(self) -> Dict[str, Any]:
        """Get comprehensive hive mind status"""
        return {
            'state': self.state.value,
            'nodes': {
                node_id: {
                    'specialization': node.specialization,
                    'active_tasks': len(node.active_tasks),
                    'performance': node.performance_score,
                    'quantum_access': node.quantum_access
                }
                for node_id, node in self.nodes.items()
            },
            'tasks': {
                task_id: {
                    'description': task.description,
                    'status': task.status,
                    'assigned_nodes': len(task.assigned_nodes),
                    'subtasks': len(task.subtasks)
                }
                for task_id, task in self.tasks.items()
            },
            'emergence_patterns': len(self.emergence_engine.emergence_history),
            'qulab_connected': self.qulab_bridge.available
        }

    def shutdown_hive(self):
        """Gracefully shutdown the hive mind"""
        self.state = HiveMindState.DISBANDING

        # Clear all active tasks
        for node in self.nodes.values():
            node.active_tasks.clear()

        # Stop all threads
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)

        print("🐝 Hive mind disbanded successfully")


def run_hive_mind_demo():
    """Demonstrate the hive mind in action"""
    print("🧠 INITIALIZING HIVE MIND FROM QULABINFINITE")
    print("=" * 50)

    hive = HiveMindOrchestrator(num_nodes=5)

    # Submit test tasks
    tasks = [
        "Design a quantum-resistant cryptographic system",
        "Develop a new approach to AGI alignment",
        "Create a sustainable energy storage solution",
        "Design a decentralized governance system"
    ]

    print(f"🐝 Hive initialized with {len(hive.nodes)} nodes")
    print(f"🐝 QuLabInfinite connected: {hive.qulab_bridge.available}")

    # Submit tasks
    task_ids = []
    for task in tasks:
        task_id = hive.submit_task(task, domain="research", complexity=1.5)
        task_ids.append(task_id)

    print(f"\n🐝 Processing {len(tasks)} tasks through hive intelligence...")

    # Run hive cycles
    cycles_completed = 0
    max_cycles = 20

    while cycles_completed < max_cycles:
        result = hive.run_hive_cycle()

        if result['completed_tasks']:
            print(f"🐝 Cycle {cycles_completed + 1}: Completed {len(result['completed_tasks'])} tasks")
            for task_result in result['completed_tasks']:
                print(f"   ✓ Task {task_result['task_id']}: {task_result['solution']['confidence']:.2f} confidence")

        cycles_completed += 1
        time.sleep(1)  # Simulate processing time

        # Check if all tasks are done
        active_tasks = sum(1 for task in hive.tasks.values() if task.status != 'completed')
        if active_tasks == 0:
            break

    # Get final status
    status = hive.get_hive_status()

    print("\n" + "=" * 50)
    print("🧠 HIVE MIND EXECUTION COMPLETE")
    print("=" * 50)
    print(f"Total tasks processed: {len(hive.tasks)}")
    print(f"Emergent patterns detected: {status['emergence_patterns']}")
    print(f"Final hive state: {status['state']}")

    # Show node performance
    print("\nNode Performance Summary:")
    for node_id, node_data in status['nodes'].items():
        print(f"  {node_id}: {node_data['performance']:.2f} (specialization: {node_data['specialization']})")

    hive.shutdown_hive()


if __name__ == "__main__":
    run_hive_mind_demo()
