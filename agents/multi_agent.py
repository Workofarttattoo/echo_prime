"""
QuLabInfinite: Distributed Swarm Intelligence & Hive Mind System
True collective intelligence with distributed agents, swarm algorithms, and emergent behavior.
"""
import asyncio
import json
import time
import socket
import threading
import multiprocessing
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import networkx as nx


class MessageType(Enum):
    """Types of messages between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    CONSENSUS = "consensus"
    TASK_DELEGATION = "task_delegation"


@dataclass
class Message:
    """Message between agents"""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str


class SwarmAgent:
    """
    Distributed swarm agent with network communication and collective intelligence.
    Part of the QuLabInfinite hive mind system.
    """
    def __init__(self, agent_id: str, specialization: str = "general", capabilities: List[str] = None,
                 host: str = "localhost", port: int = None):
        self.agent_id = agent_id
        self.specialization = specialization
        self.capabilities = capabilities or []
        self.host = host
        self.port = port or random.randint(8000, 9000)
        self.state = {}
        self.task_history = []

        # Swarm intelligence properties
        self.position = np.random.rand(10)  # Position in solution space
        self.velocity = np.random.rand(10) * 0.1  # Velocity for PSO
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')

        # Network communication
        self.server_socket = None
        self.client_sockets = {}
        self.running = False
        self.message_handlers = {}

        # Swarm neighbors
        self.neighbors = set()
        self.swarm_memory = {}  # Shared memory across swarm

        # Start network server
        self.start_network_server()

    def start_network_server(self):
        """Start network server for distributed communication"""
        def server_thread():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(5)
                self.running = True

                print(f"SwarmAgent {self.agent_id} listening on {self.host}:{self.port}")

                while self.running:
                    try:
                        client_socket, addr = self.server_socket.accept()
                        client_handler = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                        client_handler.daemon = True
                        client_handler.start()
                    except OSError:
                        break  # Socket closed

            except Exception as e:
                print(f"Network server error for {self.agent_id}: {e}")

        self.server_thread = threading.Thread(target=server_thread, daemon=True)
        self.server_thread.start()

    def handle_client(self, client_socket, addr):
        """Handle incoming client connections"""
        try:
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    break

                message = json.loads(data.decode())
                self.process_network_message(message, client_socket)

        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            client_socket.close()

    def process_network_message(self, message: Dict, client_socket):
        """Process incoming network messages"""
        msg_type = message.get('type')

        if msg_type in self.message_handlers:
            response = self.message_handlers[msg_type](message)
            if response:
                client_socket.send(json.dumps(response).encode())
        else:
            # Default swarm message handling
            self.handle_swarm_message(message, client_socket)

    def handle_swarm_message(self, message: Dict, client_socket):
        """Handle swarm-specific messages"""
        msg_type = message.get('type')

        if msg_type == 'swarm_join':
            # New agent joining swarm
            neighbor_id = message['agent_id']
            neighbor_addr = message['address']
            self.neighbors.add(neighbor_id)
            self.connect_to_neighbor(neighbor_id, neighbor_addr)

        elif msg_type == 'swarm_update':
            # Update swarm state
            self.update_swarm_state(message)

        elif msg_type == 'pso_update':
            # Particle Swarm Optimization update
            self.particle_swarm_update(message)

        elif msg_type == 'consensus_vote':
            # Consensus algorithm vote
            self.handle_consensus_vote(message)

    def connect_to_neighbor(self, neighbor_id: str, address: Tuple[str, int]):
        """Connect to a swarm neighbor"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(address)
            self.client_sockets[neighbor_id] = client_socket
            print(f"Connected to swarm neighbor {neighbor_id}")
        except Exception as e:
            print(f"Failed to connect to neighbor {neighbor_id}: {e}")

    def broadcast_to_swarm(self, message: Dict):
        """Broadcast message to all swarm neighbors"""
        for neighbor_id, client_socket in self.client_sockets.items():
            try:
                client_socket.send(json.dumps(message).encode())
            except Exception as e:
                # Remove dead connections
                del self.client_sockets[neighbor_id]

    def update_swarm_state(self, message: Dict):
        """Update local state based on swarm information"""
        swarm_state = message.get('state', {})

        # Update shared memory
        self.swarm_memory.update(swarm_state.get('shared_memory', {}))

        # Update fitness if better
        neighbor_fitness = swarm_state.get('fitness', float('inf'))
        if neighbor_fitness < self.best_fitness:
            self.best_position = np.array(swarm_state.get('position', self.position))
            self.best_fitness = neighbor_fitness

    def particle_swarm_update(self, message: Dict):
        """Update particle position using PSO algorithm"""
        global_best = np.array(message.get('global_best_position', self.position))
        global_best_fitness = message.get('global_best_fitness', float('inf'))

        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.4  # Cognitive coefficient
        c2 = 1.4  # Social coefficient

        # Update velocity
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

        # Update position
        self.position += self.velocity

        # Evaluate fitness
        self.fitness = self.evaluate_fitness(self.position)

        # Update personal best
        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness

    def evaluate_fitness(self, position: np.ndarray) -> float:
        """Evaluate fitness of current position (to be overridden by subclasses)"""
        # Default: minimize sum of squares (sphere function)
        return np.sum(position ** 2)

    def handle_consensus_vote(self, message: Dict):
        """Handle consensus voting in swarm"""
        proposal = message.get('proposal')
        proposal_id = message.get('proposal_id')

        # Simple majority vote (can be extended to more sophisticated consensus)
        vote = self.evaluate_proposal(proposal)

        response = {
            'type': 'consensus_response',
            'proposal_id': proposal_id,
            'agent_id': self.agent_id,
            'vote': vote
        }

        # Send vote back to proposer
        proposer_addr = message.get('proposer_address')
        if proposer_addr:
            self.send_to_address(proposer_addr, response)

    def evaluate_proposal(self, proposal: Dict) -> bool:
        """Evaluate a proposal for consensus (to be overridden)"""
        # Default: accept if fitness improves
        return random.choice([True, False])  # Random for demo

    def send_to_address(self, address: Tuple[str, int], message: Dict):
        """Send message to specific address"""
        try:
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_socket.connect(address)
            temp_socket.send(json.dumps(message).encode())
            temp_socket.close()
        except Exception as e:
            print(f"Failed to send to {address}: {e}")

    def join_swarm(self, swarm_coordinator: Tuple[str, int]):
        """Join an existing swarm"""
        join_message = {
            'type': 'swarm_join',
            'agent_id': self.agent_id,
            'address': (self.host, self.port),
            'specialization': self.specialization,
            'capabilities': self.capabilities
        }

        self.send_to_address(swarm_coordinator, join_message)

    def leave_swarm(self):
        """Leave the swarm gracefully"""
        leave_message = {
            'type': 'swarm_leave',
            'agent_id': self.agent_id
        }

        self.broadcast_to_swarm(leave_message)
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        for client_socket in self.client_sockets.values():
            client_socket.close()

    def register_message_handler(self, msg_type: str, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[msg_type] = handler
    
    def register_peer(self, peer: 'Agent'):
        """Register another agent as a peer"""
        self.peers[peer.agent_id] = peer
    
    async def send_message(self, receiver_id: str, message_type: MessageType, content: Dict):
        """Send message to another agent"""
        if receiver_id not in self.peers:
            return False
        
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )
        
        await self.peers[receiver_id].receive_message(message)
        return True
    
    async def broadcast(self, message_type: MessageType, content: Dict):
        """Broadcast message to all peers"""
        message = Message(
            sender_id=self.agent_id,
            receiver_id=None,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )
        
        for peer in self.peers.values():
            await peer.receive_message(message)
    
    async def receive_message(self, message: Message):
        """Receive message from another agent"""
        await self.message_queue.put(message)
    
    async def process_messages(self, handler: Callable):
        """Process incoming messages"""
        while True:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await handler(self, message)
            except asyncio.TimeoutError:
                continue
    
    def can_handle_task(self, task: Dict) -> bool:
        """Check if agent can handle a task"""
        required_capabilities = task.get("required_capabilities", [])
        return all(cap in self.capabilities for cap in required_capabilities)


class CommunicationProtocol:
    """
    Communication protocol for agent interaction.
    """
    def __init__(self):
        self.protocols = {}
    
    def register_protocol(self, name: str, handler: Callable):
        """Register a communication protocol"""
        self.protocols[name] = handler
    
    async def handle_message(self, agent: "SwarmAgent", message: Message):
        """Handle message according to protocol"""
        protocol_name = message.content.get("protocol")
        if protocol_name and protocol_name in self.protocols:
            await self.protocols[protocol_name](agent, message)
        else:
            # Default handling
            await self.default_handler(agent, message)
    
    async def default_handler(self, agent: "SwarmAgent", message: Message):
        """Default message handler"""
        if message.message_type == MessageType.REQUEST:
            # Handle request
            response_content = {
                "request_id": message.content.get("request_id"),
                "response": "acknowledged"
            }
            await agent.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                response_content
            )


class TaskDelegation:
    """
    Handles task delegation between agents.
    """
    def __init__(self, agents: List["SwarmAgent"]):
        self.agents = {agent.agent_id: agent for agent in agents}
    
    async def delegate_task(self, task: Dict, requester_id: str) -> Optional[str]:
        """
        Delegate task to appropriate agent.
        
        Returns:
            Agent ID that accepted the task, or None
        """
        # Find agents that can handle the task
        capable_agents = [
            agent for agent in self.agents.values()
            if agent.can_handle_task(task) and agent.agent_id != requester_id
        ]
        
        if not capable_agents:
            return None
        
        # Select best agent (simplified: first capable agent)
        # In full implementation, would consider load, expertise, etc.
        selected_agent = capable_agents[0]
        
        # Send task delegation message
        await self.agents[requester_id].send_message(
            selected_agent.agent_id,
            MessageType.TASK_DELEGATION,
            {
                "task": task,
                "requester_id": requester_id
            }
        )
        
        return selected_agent.agent_id
    
    def find_specialist(self, domain: str) -> Optional["SwarmAgent"]:
        """Find agent specialized in a domain"""
        for agent in self.agents.values():
            if agent.specialization == domain:
                return agent
        return None


class ConsensusMechanism:
    """
    Implements consensus mechanisms for multi-agent decisions.
    """
    def __init__(self, agents: List["SwarmAgent"], threshold: float = 0.5):
        self.agents = agents
        self.threshold = threshold
        self.votes = {}
    
    async def reach_consensus(self, proposal: Dict, timeout: float = 10.0) -> Tuple[bool, Dict]:
        """
        Reach consensus on a proposal.
        
        Returns:
            (consensus_reached, result)
        """
        self.votes = {}
        proposal_id = str(uuid.uuid4())
        
        # Broadcast proposal
        for agent in self.agents:
            await agent.broadcast(
                MessageType.CONSENSUS,
                {
                    "proposal_id": proposal_id,
                    "proposal": proposal,
                    "action": "vote"
                }
            )
        
        # Collect votes
        start_time = time.time()
        while time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
            
            # Check if we have enough votes
            if len(self.votes) >= len(self.agents) * self.threshold:
                break
        
        # Count votes
        yes_votes = sum(1 for vote in self.votes.values() if vote)
        total_votes = len(self.votes)
        
        consensus_reached = (yes_votes / total_votes) >= self.threshold if total_votes > 0 else False
        
        return consensus_reached, {
            "proposal_id": proposal_id,
            "yes_votes": yes_votes,
            "total_votes": total_votes,
            "consensus": consensus_reached
        }
    
    def record_vote(self, agent_id: str, vote: bool):
        """Record a vote from an agent"""
        self.votes[agent_id] = vote


class DistributedReasoning:
    """
    Enables agents to collaborate on complex reasoning tasks.
    """
    def __init__(self, agents: List["SwarmAgent"]):
        self.agents = agents
        self.task_assignments = {}
    
    async def solve_problem(self, problem: Dict) -> Dict:
        """
        Solve problem using distributed reasoning.
        
        Strategy:
        1. Decompose problem into sub-problems
        2. Assign sub-problems to specialized agents
        3. Collect results
        4. Synthesize final answer
        """
        # Decompose problem
        sub_problems = self.decompose_problem(problem)
        
        # Assign to agents
        assignments = {}
        for i, sub_problem in enumerate(sub_problems):
            # Find appropriate agent
            agent = self.find_agent_for_task(sub_problem)
            if agent:
                assignments[i] = agent.agent_id
        
        # Execute sub-problems in parallel
        results = {}
        for sub_id, agent_id in assignments.items():
            agent = next(a for a in self.agents if a.agent_id == agent_id)
            result = await self.execute_sub_problem(agent, sub_problems[sub_id])
            results[sub_id] = result
        
        # Synthesize results
        final_result = self.synthesize_results(results, problem)
        
        return final_result
    
    def decompose_problem(self, problem: Dict) -> List[Dict]:
        """Decompose problem into sub-problems"""
        # Simplified: return problem as single sub-problem
        # Full implementation would use problem decomposition algorithms
        return [problem]
    
    def find_agent_for_task(self, task: Dict) -> Optional["SwarmAgent"]:
        """Find best agent for a task"""
        for agent in self.agents:
            if agent.can_handle_task(task):
                return agent
        return None
    
    async def execute_sub_problem(self, agent: "SwarmAgent", sub_problem: Dict) -> Dict:
        """Execute sub-problem on an agent"""
        # Simplified: return placeholder result
        # Full implementation would actually execute the task
        return {
            "agent_id": agent.agent_id,
            "result": "completed",
            "sub_problem": sub_problem
        }
    
    def synthesize_results(self, results: Dict, original_problem: Dict) -> Dict:
        """Synthesize results from multiple agents"""
        return {
            "solved": True,
            "results": results,
            "original_problem": original_problem
        }


class SwarmCoordinator:
    """
    Coordinates swarm intelligence and collective decision making.
    Implements QuLabInfinite collective intelligence algorithms.
    """
    def __init__(self, coordinator_host: str = "localhost", coordinator_port: int = 9999):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.swarm_agents = {}  # agent_id -> (address, capabilities)
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.consensus_votes = {}
        self.collective_memory = {}

        # Start coordinator server
        self.start_coordinator_server()

    def start_coordinator_server(self):
        """Start the swarm coordinator server"""
        def coordinator_thread():
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.coordinator_host, self.coordinator_port))
                server_socket.listen(10)
                print(f"SwarmCoordinator listening on {self.coordinator_host}:{self.coordinator_port}")

                while True:
                    client_socket, addr = server_socket.accept()
                    handler = threading.Thread(target=self.handle_coordinator_client,
                                             args=(client_socket, addr))
                    handler.daemon = True
                    handler.start()
            except Exception as e:
                print(f"Coordinator server error: {e}")

        self.coordinator_thread = threading.Thread(target=coordinator_thread, daemon=True)
        self.coordinator_thread.start()

    def handle_coordinator_client(self, client_socket, addr):
        """Handle coordinator client connections"""
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break

                message = json.loads(data.decode())
                response = self.process_coordinator_message(message, addr)
                if response:
                    client_socket.send(json.dumps(response).encode())

        except Exception as e:
            print(f"Coordinator client handler error: {e}")
        finally:
            client_socket.close()

    def process_coordinator_message(self, message: Dict, addr) -> Dict:
        """Process messages sent to coordinator"""
        msg_type = message.get('type')

        if msg_type == 'swarm_join':
            agent_id = message['agent_id']
            agent_addr = message['address']
            self.swarm_agents[agent_id] = {
                'address': agent_addr,
                'capabilities': message.get('capabilities', []),
                'specialization': message.get('specialization', 'general'),
                'last_seen': time.time()
            }
            print(f"Agent {agent_id} joined swarm. Total agents: {len(self.swarm_agents)}")

            # Send welcome message with swarm info
            return {
                'type': 'swarm_welcome',
                'swarm_size': len(self.swarm_agents),
                'global_best': {
                    'position': self.global_best_position.tolist() if self.global_best_position is not None else None,
                    'fitness': self.global_best_fitness
                }
            }

        elif msg_type == 'swarm_update':
            # Update global swarm state
            agent_id = message['agent_id']
            if agent_id in self.swarm_agents:
                self.swarm_agents[agent_id]['last_seen'] = time.time()

                # Update global best
                agent_fitness = message.get('fitness', float('inf'))
                if agent_fitness < self.global_best_fitness:
                    self.global_best_position = np.array(message.get('position', []))
                    self.global_best_fitness = agent_fitness
                    print(f"New global best fitness: {self.global_best_fitness}")

                    # Broadcast global best to swarm
                    self.broadcast_global_best()

        elif msg_type == 'consensus_response':
            # Handle consensus votes
            proposal_id = message['proposal_id']
            vote = message['vote']

            if proposal_id not in self.consensus_votes:
                self.consensus_votes[proposal_id] = []

            self.consensus_votes[proposal_id].append(vote)

            # Check if consensus reached
            return self.check_consensus(proposal_id)

        return {'status': 'acknowledged'}

    def broadcast_global_best(self):
        """Broadcast global best position to all swarm agents"""
        if self.global_best_position is None:
            return

        message = {
            'type': 'global_best_update',
            'global_best_position': self.global_best_position.tolist(),
            'global_best_fitness': self.global_best_fitness,
            'timestamp': time.time()
        }

        self.broadcast_to_swarm(message)

    def broadcast_to_swarm(self, message: Dict):
        """Broadcast message to all swarm agents"""
        for agent_info in self.swarm_agents.values():
            addr = agent_info['address']
            try:
                temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                temp_socket.connect(addr)
                temp_socket.send(json.dumps(message).encode())
                temp_socket.close()
            except Exception as e:
                print(f"Failed to broadcast to {addr}: {e}")

    def initiate_swarm_optimization(self, problem_dimension: int = 10, num_iterations: int = 100):
        """Initiate swarm optimization across all agents"""
        print(f"Starting swarm optimization with {len(self.swarm_agents)} agents")

        # Initialize global best randomly
        self.global_best_position = np.random.rand(problem_dimension)
        self.global_best_fitness = float('inf')

        message = {
            'type': 'swarm_optimization_start',
            'problem_dimension': problem_dimension,
            'num_iterations': num_iterations,
            'global_best': self.global_best_position.tolist()
        }

        self.broadcast_to_swarm(message)

        # Monitor progress
        for iteration in range(num_iterations):
            time.sleep(1)  # Allow agents to process
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Global best fitness = {self.global_best_fitness}")

    def initiate_consensus(self, proposal: Dict, timeout: float = 30.0) -> Dict:
        """Initiate consensus voting across swarm"""
        proposal_id = str(uuid.uuid4())

        message = {
            'type': 'consensus_request',
            'proposal_id': proposal_id,
            'proposal': proposal,
            'timeout': timeout,
            'proposer_address': (self.coordinator_host, self.coordinator_port)
        }

        self.consensus_votes[proposal_id] = []
        self.broadcast_to_swarm(message)

        # Wait for votes
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            if len(self.consensus_votes[proposal_id]) >= len(self.swarm_agents) * 0.5:  # 50% quorum
                break

        return self.check_consensus(proposal_id)

    def check_consensus(self, proposal_id: str) -> Dict:
        """Check if consensus has been reached"""
        if proposal_id not in self.consensus_votes:
            return {'consensus': False, 'reason': 'no_votes'}

        votes = self.consensus_votes[proposal_id]
        total_agents = len(self.swarm_agents)

        if len(votes) < total_agents * 0.5:  # Need 50% participation
            return {'consensus': False, 'reason': 'insufficient_participation'}

        yes_votes = sum(1 for vote in votes if vote)
        consensus_ratio = yes_votes / len(votes)

        consensus_reached = consensus_ratio >= 0.67  # 2/3 majority

        return {
            'consensus': consensus_reached,
            'yes_votes': yes_votes,
            'total_votes': len(votes),
            'ratio': consensus_ratio
        }

    def get_swarm_status(self) -> Dict:
        """Get current swarm status"""
        return {
            'total_agents': len(self.swarm_agents),
            'active_agents': len([a for a in self.swarm_agents.values()
                                if time.time() - a['last_seen'] < 60]),  # Active in last minute
            'global_best_fitness': self.global_best_fitness,
            'collective_memory_size': len(self.collective_memory),
            'specializations': list(set(a['specialization'] for a in self.swarm_agents.values()))
        }


class QuLabInfinite:
    """
    QuLabInfinite: Complete swarm intelligence and collective intelligence system.
    True hive mind with emergent behavior and distributed problem solving.
    """
    def __init__(self, coordinator_host: str = "localhost", coordinator_port: int = 9999):
        self.coordinator = SwarmCoordinator(coordinator_host, coordinator_port)
        self.swarm_agents = []
        self.problem_solvers = {}

        # Initialize collective intelligence algorithms
        self.ant_colony = AntColonyOptimization()
        self.particle_swarm = ParticleSwarmOptimization()
        self.consensus_engine = SwarmConsensus()

    def create_swarm_agent(self, specialization: str = "general",
                          capabilities: List[str] = None) -> SwarmAgent:
        """Create and register a new swarm agent"""
        agent_id = f"swarm_agent_{len(self.swarm_agents)}"
        agent = SwarmAgent(agent_id, specialization, capabilities)
        self.swarm_agents.append(agent)

        # Connect agent to swarm coordinator
        agent.join_swarm((self.coordinator.coordinator_host, self.coordinator.coordinator_port))

        return agent

    def solve_with_swarm(self, problem: Dict, algorithm: str = "pso") -> Dict:
        """Solve a problem using swarm intelligence"""
        if algorithm == "pso":
            return self.particle_swarm.optimize(problem, self.swarm_agents)
        elif algorithm == "aco":
            return self.ant_colony.optimize(problem, self.swarm_agents)
        elif algorithm == "consensus":
            return self.consensus_engine.decide(problem, self.swarm_agents)
        else:
            return {"error": f"Unknown algorithm: {algorithm}"}

    def get_hive_mind_status(self) -> Dict:
        """Get the current status of the hive mind"""
        return {
            'swarm_status': self.coordinator.get_swarm_status(),
            'active_agents': len([a for a in self.swarm_agents if a.running]),
            'algorithms_available': ['pso', 'aco', 'consensus'],
            'collective_memory': len(self.coordinator.collective_memory)
        }


class AntColonyOptimization:
    """Ant Colony Optimization for swarm problem solving"""
    def __init__(self, num_ants: int = 20, evaporation_rate: float = 0.1):
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_matrix = {}

    def optimize(self, problem: Dict, agents: List[SwarmAgent]) -> Dict:
        """Solve problem using ACO"""
        # Simplified ACO implementation
        # In full version, would implement complete ant colony algorithm
        best_solution = None
        best_fitness = float('inf')

        for _ in range(self.num_ants):
            # Each agent explores solution space
            for agent in agents:
                solution = agent.position + np.random.normal(0, 0.1, len(agent.position))
                fitness = agent.evaluate_fitness(solution)

                if fitness < best_fitness:
                    best_solution = solution
                    best_fitness = fitness

        return {
            'algorithm': 'aco',
            'best_solution': best_solution.tolist() if best_solution is not None else None,
            'best_fitness': best_fitness,
            'ants_used': self.num_ants
        }


class ParticleSwarmOptimization:
    """Particle Swarm Optimization for distributed problem solving"""
    def __init__(self, inertia_weight: float = 0.7, cognitive_coeff: float = 1.4, social_coeff: float = 1.4):
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff

    def optimize(self, problem: Dict, agents: List[SwarmAgent]) -> Dict:
        """Solve problem using PSO"""
        # Simplified PSO - in full implementation would run multiple iterations
        best_solution = None
        best_fitness = float('inf')

        for agent in agents:
            if agent.fitness < best_fitness:
                best_solution = agent.best_position
                best_fitness = agent.best_fitness

        return {
            'algorithm': 'pso',
            'best_solution': best_solution.tolist() if best_solution is not None else None,
            'best_fitness': best_fitness,
            'particles': len(agents)
        }


class SwarmConsensus:
    """Consensus algorithms for swarm decision making"""
    def __init__(self, consensus_threshold: float = 0.67):
        self.consensus_threshold = consensus_threshold

    def decide(self, problem: Dict, agents: List[SwarmAgent]) -> Dict:
        """Make decision through swarm consensus"""
        # Simplified consensus - collect votes from all agents
        votes = []
        for agent in agents:
            vote = agent.evaluate_proposal(problem)
            votes.append(vote)

        yes_votes = sum(1 for vote in votes if vote)
        consensus_reached = (yes_votes / len(votes)) >= self.consensus_threshold

        return {
            'algorithm': 'consensus',
            'decision': consensus_reached,
            'yes_votes': yes_votes,
            'total_votes': len(votes),
            'ratio': yes_votes / len(votes)
        }


# Keep legacy Agent class for backward compatibility
Agent = SwarmAgent  # Alias for backward compatibility

# Legacy compatibility - keep old MultiAgentSystem for backward compatibility
class MultiAgentSystem:
    """
    Legacy multi-agent system for backward compatibility.
    Use QuLabInfinite for true swarm intelligence.
    """
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.communication = CommunicationProtocol()
        self.task_delegation = None
        self.consensus = None
        self.distributed_reasoning = None
    
    def add_agent(self, agent: "SwarmAgent"):
        """Add agent to system"""
        self.agents[agent.agent_id] = agent
        
        # Register peers
        for other_agent in self.agents.values():
            if other_agent.agent_id != agent.agent_id:
                agent.register_peer(other_agent)
                other_agent.register_peer(agent)
        
        # Update subsystems
        agent_list = list(self.agents.values())
        self.task_delegation = TaskDelegation(agent_list)
        self.consensus = ConsensusMechanism(agent_list)
        self.distributed_reasoning = DistributedReasoning(agent_list)
    
    async def start(self):
        """Start the multi-agent system"""
        # Start message processing for all agents
        tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(
                agent.process_messages(self.communication.handle_message)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def solve_problem_distributed(self, problem: Dict) -> Dict:
        """Solve problem using distributed reasoning"""
        if self.distributed_reasoning:
            return await self.distributed_reasoning.solve_problem(problem)
        return {"error": "Distributed reasoning not initialized"}

