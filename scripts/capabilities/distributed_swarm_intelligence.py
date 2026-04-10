"""
ECH0-PRIME: Distributed Swarm Intelligence System
Implements advanced multi-agent coordination, emergent intelligence, and fault-tolerant distributed processing.
"""

import asyncio
import json
import time
import uuid
import random
import logging
import threading
import socket
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SwarmIntelligence")

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"

class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class SwarmTask:
    task_id: str
    description: str
    priority: TaskPriority
    required_capabilities: List[str]
    data: Dict[str, Any]
    assigned_agent: Optional[str] = None
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    timeout: float = 30.0

@dataclass
class AgentMetadata:
    agent_id: str
    capabilities: List[str]
    specialization: str
    status: AgentStatus = AgentStatus.OFFLINE
    last_heartbeat: float = 0.0
    load_score: float = 0.0
    reliability_score: float = 1.0

class DistributedSwarmIntelligence:
    """
    Advanced Distributed Swarm Intelligence System for ECH0-PRIME.
    Coordinates multiple agents to solve complex problems through emergent behavior.
    """
    def __init__(self, coordinator_port: int = 10000):
        self.system_id = f"swarm_sys_{uuid.uuid4().hex[:8]}"
        self.coordinator_port = coordinator_port
        self.agents: Dict[str, AgentMetadata] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.task_queue = deque()
        
        # Swarm Intelligence Properties
        self.global_best_fitness = float('inf')
        self.global_best_solution = None
        self.collective_knowledge = {}
        self.emergence_patterns = []
        
        # Communication
        self.running = False
        self.server_thread = None
        self.health_monitor_thread = None
        
        # Lock for thread safety
        self.lock = threading.Lock()

    def start(self):
        """Start the swarm coordinator and health monitor"""
        self.running = True
        self.server_thread = threading.Thread(target=self._run_coordinator_server, daemon=True)
        self.server_thread.start()
        
        self.health_monitor_thread = threading.Thread(target=self._monitor_agent_health, daemon=True)
        self.health_monitor_thread.start()
        
        logger.info(f"Swarm Intelligence System {self.system_id} started on port {self.coordinator_port}")

    def stop(self):
        """Stop the swarm system"""
        self.running = False
        logger.info("Swarm Intelligence System shutting down...")

    def _run_coordinator_server(self):
        """Internal server to listen for agent registrations and updates"""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', self.coordinator_port))
            server.listen(5)
            server.settimeout(1.0)
            
            while self.running:
                try:
                    conn, addr = server.accept()
                    client_thread = threading.Thread(target=self._handle_agent_connection, args=(conn, addr), daemon=True)
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Coordinator server accept error: {e}")
        except Exception as e:
            logger.error(f"Coordinator server failed to start: {e}")

    def _handle_agent_connection(self, conn, addr):
        """Handle individual agent communication"""
        try:
            data = conn.recv(4096)
            if not data:
                return
            
            message = json.loads(data.decode())
            msg_type = message.get('type')
            
            response = {"status": "error", "message": "unknown_type"}
            
            if msg_type == 'register':
                response = self._register_agent(message)
            elif msg_type == 'heartbeat':
                response = self._process_heartbeat(message)
            elif msg_type == 'task_update':
                response = self._process_task_update(message)
            elif msg_type == 'find_peers':
                response = self._get_active_peers(message)
                
            conn.send(json.dumps(response).encode())
        except Exception as e:
            logger.error(f"Error handling agent connection: {e}")
        finally:
            conn.close()

    def _register_agent(self, data: Dict) -> Dict:
        """Register a new agent in the swarm"""
        agent_id = data.get('agent_id')
        with self.lock:
            self.agents[agent_id] = AgentMetadata(
                agent_id=agent_id,
                capabilities=data.get('capabilities', []),
                specialization=data.get('specialization', 'general'),
                status=AgentStatus.IDLE,
                last_heartbeat=time.time()
            )
        logger.info(f"Agent {agent_id} registered with capabilities: {data.get('capabilities')}")
        return {"status": "success", "system_id": self.system_id}

    def _process_heartbeat(self, data: Dict) -> Dict:
        """Update agent status and load"""
        agent_id = data.get('agent_id')
        with self.lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.last_heartbeat = time.time()
                agent.load_score = data.get('load', 0.0)
                agent.status = AgentStatus(data.get('status', 'idle'))
                return {"status": "success"}
        return {"status": "error", "message": "agent_not_found"}

    def _process_task_update(self, data: Dict) -> Dict:
        """Process updates on task progress or completion"""
        task_id = data.get('task_id')
        agent_id = data.get('agent_id')
        
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = data.get('status', task.status)
                task.progress = data.get('progress', task.progress)
                task.result = data.get('result', task.result)
                
                if task.status == 'completed':
                    logger.info(f"Task {task_id} completed by agent {agent_id}")
                    # Update swarm intelligence from result
                    self._update_collective_intelligence(task)
                
                return {"status": "success"}
        return {"status": "error", "message": "task_not_found"}

    def _get_active_peers(self, data: Dict) -> Dict:
        """Return list of other active agents for peer-to-peer collaboration"""
        requestor_id = data.get('agent_id')
        peers = []
        with self.lock:
            for aid, agent in self.agents.items():
                if aid != requestor_id and agent.status != AgentStatus.OFFLINE:
                    peers.append({
                        "agent_id": aid,
                        "specialization": agent.specialization,
                        "capabilities": agent.capabilities
                    })
        return {"status": "success", "peers": peers}

    def _monitor_agent_health(self):
        """Periodically check for offline agents and reassign tasks"""
        while self.running:
            try:
                now = time.time()
                with self.lock:
                    for agent_id, agent in self.agents.items():
                        if agent.status != AgentStatus.OFFLINE and now - agent.last_heartbeat > 10.0:
                            logger.warning(f"Agent {agent_id} timed out, marking offline")
                            agent.status = AgentStatus.OFFLINE
                            self._handle_agent_failure(agent_id)
                
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def _handle_agent_failure(self, agent_id: str):
        """Reassign tasks from a failed agent"""
        for task_id, task in self.tasks.items():
            if task.assigned_agent == agent_id and task.status != 'completed':
                logger.info(f"Reassigning task {task_id} from failed agent {agent_id}")
                task.status = 'pending'
                task.assigned_agent = None
                self.task_queue.appendleft(task_id)

    def submit_task(self, description: str, priority: TaskPriority = TaskPriority.MEDIUM, 
                    capabilities: List[str] = None, data: Dict = None) -> str:
        """Submit a new task to the swarm"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = SwarmTask(
            task_id=task_id,
            description=description,
            priority=priority,
            required_capabilities=capabilities or [],
            data=data or {}
        )
        
        with self.lock:
            self.tasks[task_id] = task
            self.task_queue.append(task_id)
            
        logger.info(f"Submitted task {task_id}: {description}")
        return task_id

    def coordinate_swarm(self):
        """The main coordination cycle: match tasks to agents"""
        with self.lock:
            if not self.task_queue:
                return
            
            # Sort tasks by priority
            self.task_queue = deque(sorted(self.task_queue, key=lambda tid: self.tasks[tid].priority.value, reverse=True))
            
            processed_tasks = []
            while self.task_queue:
                task_id = self.task_queue.popleft()
                task = self.tasks[task_id]
                
                # Find best agent
                best_agent = self._find_best_agent(task)
                
                if best_agent:
                    self._assign_task(task, best_agent)
                    processed_tasks.append(task_id)
                else:
                    # No agent available right now, put back in queue
                    self.task_queue.append(task_id)
                    break 

    def _find_best_agent(self, task: SwarmTask) -> Optional[str]:
        """Find the most suitable idle agent for a task"""
        best_agent_id = None
        best_score = -1.0
        
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.IDLE:
                # Calculate compatibility score
                cap_match = sum(1 for cap in task.required_capabilities if cap in agent.capabilities)
                score = (cap_match * 2.0) + agent.reliability_score - (agent.load_score * 0.5)
                
                if score > best_score:
                    best_score = score
                    best_agent_id = agent_id
                    
        return best_agent_id

    def _assign_task(self, task: SwarmTask, agent_id: str):
        """Assign task to agent and notify (simulated here)"""
        task.assigned_agent = agent_id
        task.status = 'assigned'
        self.agents[agent_id].status = AgentStatus.BUSY
        logger.info(f"Assigned task {task.task_id} to agent {agent_id}")
        # In a real networked scenario, we would send a message to the agent here

    def _update_collective_intelligence(self, task: SwarmTask):
        """Update swarm knowledge based on task results"""
        if task.result:
            # Simple example: update a collective value if found
            if isinstance(task.result, dict) and 'fitness' in task.result:
                fitness = task.result['fitness']
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = task.result.get('solution')
                    logger.info(f"NEW SWARM GLOBAL BEST: {fitness}")
            
            # Detect emergent patterns
            self._detect_emergence(task)

    def _detect_emergence(self, task: SwarmTask):
        """Heuristic for detecting emergent patterns in problem solving"""
        # Look for repeated solution structures or cross-agent improvements
        pattern = {
            "timestamp": time.time(),
            "task_id": task.task_id,
            "agent_id": task.assigned_agent,
            "insight": "potential_pattern_detected"
        }
        self.emergence_patterns.append(pattern)

    def get_swarm_status(self) -> Dict:
        """Get comprehensive status of the swarm"""
        with self.lock:
            return {
                "system_id": self.system_id,
                "total_agents": len(self.agents),
                "active_agents": sum(1 for a in self.agents.values() if a.status != AgentStatus.OFFLINE),
                "busy_agents": sum(1 for a in self.agents.values() if a.status == AgentStatus.BUSY),
                "pending_tasks": len(self.task_queue),
                "completed_tasks": sum(1 for t in self.tasks.values() if t.status == 'completed'),
                "global_best_fitness": self.global_best_fitness,
                "emergence_patterns": len(self.emergence_patterns)
            }

class SwarmWorker:
    """
    Simulated agent that can join the ECH0-PRIME swarm.
    In production, these would be separate processes/containers.
    """
    def __init__(self, agent_id: str, coordinator_host: str, coordinator_port: int, 
                 capabilities: List[str], specialization: str):
        self.agent_id = agent_id
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.capabilities = capabilities
        self.specialization = specialization
        self.status = AgentStatus.IDLE
        self.running = False
        self.current_task = None

    def join(self):
        """Register with the coordinator"""
        try:
            self._send_to_coordinator({
                "type": "register",
                "agent_id": self.agent_id,
                "capabilities": self.capabilities,
                "specialization": self.specialization
            })
            self.running = True
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
            logger.info(f"Worker {self.agent_id} joined swarm")
        except Exception as e:
            logger.error(f"Worker {self.agent_id} failed to join: {e}")

    def _send_to_coordinator(self, message: Dict) -> Dict:
        """Utility to send messages to the coordinator"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.coordinator_host, self.coordinator_port))
            s.sendall(json.dumps(message).encode())
            data = s.recv(4096)
            return json.loads(data.decode())

    def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                self._send_to_coordinator({
                    "type": "heartbeat",
                    "agent_id": self.agent_id,
                    "status": self.status.value,
                    "load": 1.0 if self.current_task else 0.0
                })
                time.sleep(3.0)
            except Exception:
                # Coordinator might be down, keep trying
                time.sleep(5.0)

    def simulate_task_work(self, task_id: str, duration: float = 2.0):
        """Simulate performing a task"""
        self.status = AgentStatus.BUSY
        self.current_task = task_id
        
        def work():
            time.sleep(duration)
            result = {
                "fitness": random.random() * 100,
                "solution": f"Result for {task_id} by {self.agent_id}"
            }
            try:
                self._send_to_coordinator({
                    "type": "task_update",
                    "task_id": task_id,
                    "agent_id": self.agent_id,
                    "status": "completed",
                    "progress": 1.0,
                    "result": result
                })
            except Exception:
                pass
            self.status = AgentStatus.IDLE
            self.current_task = None
            
        threading.Thread(target=work, daemon=True).start()

def demonstrate_swarm():
    """Simple demo of the swarm intelligence system"""
    print("\n--- ECH0-PRIME Distributed Swarm Intelligence Demo ---")
    
    # Start Coordinator
    coordinator = DistributedSwarmIntelligence(coordinator_port=11000)
    coordinator.start()
    
    # Create some workers
    workers = [
        SwarmWorker("agent_alpha", "localhost", 11000, ["math", "logic"], "mathematician"),
        SwarmWorker("agent_beta", "localhost", 11000, ["vision", "spatial"], "architect"),
        SwarmWorker("agent_gamma", "localhost", 11000, ["code", "optimization"], "engineer")
    ]
    
    for worker in workers:
        worker.join()
        
    # Submit tasks
    coordinator.submit_task("Calculate optimal trajectory", TaskPriority.HIGH, ["math"])
    coordinator.submit_task("Analyze grid pattern for ARC", TaskPriority.MEDIUM, ["vision"])
    coordinator.submit_task("Optimize GPU kernels", TaskPriority.CRITICAL, ["code"])
    
    # Run coordination
    for _ in range(5):
        coordinator.coordinate_swarm()
        
        # Simulate workers picking up assigned tasks
        with coordinator.lock:
            for tid, task in coordinator.tasks.items():
                if task.status == 'assigned' and task.assigned_agent:
                    # Find the worker and tell it to work (in reality, the worker would poll or be pushed)
                    for w in workers:
                        if w.agent_id == task.assigned_agent and w.status == AgentStatus.IDLE:
                            w.simulate_task_work(tid)
                            task.status = 'in_progress'
                            
        time.sleep(2)
        status = coordinator.get_swarm_status()
        print(f"Swarm Status: {status['active_agents']} active, {status['busy_agents']} busy, {status['pending_tasks']} pending, {status['completed_tasks']} completed")

    coordinator.stop()
    print("--- Demo Complete ---\n")

if __name__ == "__main__":
    demonstrate_swarm()



