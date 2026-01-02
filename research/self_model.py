"""
Integrated Information Theory (IIT) and enhanced Global Workspace Theory.
"""
import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Tuple
from scipy.linalg import logm, expm
from scipy.stats import entropy


class IntegratedInformationTheory:
    """
    Implements Integrated Information Theory (IIT) 3.0 for consciousness measurement.
    Computes Phi (Φ) - the amount of integrated information using proper IIT formalism.
    """
    def __init__(self):
        self.earth_mover_distance_cache = {}

    def compute_phi(self, system_state: np.ndarray, partition: Optional[Tuple] = None) -> float:
        """
        Compute Phi (Φ) - integrated information using IIT 3.0 formalism.

        Phi measures the distance between the actual system and its minimum information partition.
        Uses Earth Mover's Distance between cause-effect repertoires.
        """
        if partition is None:
            # Find minimum information partition (MIP)
            partition = self.find_mip(system_state)

        # Compute cause-effect repertoires
        repertoire_whole = self.compute_cause_effect_repertoire(system_state)
        repertoire_partition = self.compute_partitioned_repertoire(system_state, partition)

        # Phi is the Earth Mover's Distance between repertoires
        phi = self.earth_movers_distance(repertoire_whole, repertoire_partition)

        return max(0.0, phi)

    def find_mip(self, system_state: np.ndarray) -> Tuple:
        """
        Find Minimum Information Partition (MIP) using IIT formalism.
        The partition that minimizes integrated information Φ.
        """
        n = len(system_state)
        if n <= 1:
            return ([0], []) if n == 1 else ([], [])

        min_phi = float('inf')
        best_partition = None

        # Try all possible bipartitions (exponential complexity - optimize for small systems)
        for i in range(1, n//2 + 1):
            # Try different partition sizes
            for partition_size in range(1, n//2 + 1):
                partition = (list(range(partition_size)), list(range(partition_size, n)))
                phi = self.compute_phi(system_state, partition)
                if phi < min_phi:
                    min_phi = phi
                    best_partition = partition

        return best_partition or (list(range(n//2)), list(range(n//2, n)))

    def compute_cause_effect_repertoire(self, system_state: np.ndarray) -> Dict:
        """
        Compute the cause-effect repertoire of the system.
        This represents what the system specifies about causes and effects.
        """
        # Convert system state to TPM (Transition Probability Matrix)
        tpm = self.state_to_tpm(system_state)

        # Compute cause repertoire (what causes this state)
        cause_repertoire = self.compute_cause_repertoire(tpm)

        # Compute effect repertoire (what this state causes)
        effect_repertoire = self.compute_effect_repertoire(tpm)

        return {
            'cause': cause_repertoire,
            'effect': effect_repertoire,
            'tpm': tpm
        }

    def compute_partitioned_repertoire(self, system_state: np.ndarray, partition: Tuple) -> Dict:
        """
        Compute repertoire of partitioned system.
        """
        part1, part2 = partition

        # Convert to numpy array if needed and index properly
        state_array = np.array(system_state)
        repertoire_part1 = self.compute_cause_effect_repertoire(state_array[part1])
        repertoire_part2 = self.compute_cause_effect_repertoire(state_array[part2])

        # Combine repertoires (in IIT, this is the "unified" repertoire of parts)
        return self.combine_repertoires([repertoire_part1, repertoire_part2])

    def state_to_tpm(self, system_state: np.ndarray) -> np.ndarray:
        """
        Convert system state to Transition Probability Matrix.
        For simplicity, treat as a small discrete system.
        """
        n = len(system_state)

        # Discretize state into 2^n possible states (for small n)
        if n > 10:
            # For large systems, use approximation
            tpm = np.outer(system_state, system_state)
            tpm = tpm / tpm.sum(axis=1, keepdims=True)
        else:
            # Exact TPM for small systems
            num_states = 2 ** n
            tpm = np.zeros((num_states, num_states))

            # Simple deterministic transitions based on state
            for i in range(num_states):
                for j in range(num_states):
                    # Transition probability based on Hamming distance
                    state_i = [(i >> k) & 1 for k in range(n)]
                    state_j = [(j >> k) & 1 for k in range(n)]
                    distance = sum(a != b for a, b in zip(state_i, state_j))
                    tpm[i, j] = np.exp(-distance)  # Higher probability for similar states

                tpm[i] /= tpm[i].sum()

        return tpm

    def compute_cause_repertoire(self, tpm: np.ndarray) -> np.ndarray:
        """
        Compute cause repertoire: what causes each state.
        """
        # Cause repertoire is the marginal distribution over causes
        # In IIT, this is P(cause|effect) marginalized over effects
        cause_repertoire = np.sum(tpm, axis=1)  # Sum over effect dimension
        cause_repertoire /= cause_repertoire.sum()

        return cause_repertoire

    def compute_effect_repertoire(self, tpm: np.ndarray) -> np.ndarray:
        """
        Compute effect repertoire: what each state causes.
        """
        # Effect repertoire is the marginal distribution over effects
        effect_repertoire = np.sum(tpm, axis=0)  # Sum over cause dimension
        effect_repertoire /= effect_repertoire.sum()

        return effect_repertoire

    def combine_repertoires(self, repertoires: List[Dict]) -> Dict:
        """
        Combine multiple repertoires (for partitioned systems).
        """
        combined_cause = np.zeros_like(repertoires[0]['cause'])
        combined_effect = np.zeros_like(repertoires[0]['effect'])

        for repertoire in repertoires:
            combined_cause += repertoire['cause']
            combined_effect += repertoire['effect']

        # Average the repertoires
        combined_cause /= len(repertoires)
        combined_effect /= len(repertoires)

        return {
            'cause': combined_cause,
            'effect': combined_effect,
            'tpm': None  # Combined TPM would be more complex
        }

    def earth_movers_distance(self, repertoire1: Dict, repertoire2: Dict) -> float:
        """
        Compute Earth Mover's Distance between two repertoires.
        This is the IIT measure of integrated information Φ.
        """
        cause1, cause2 = repertoire1['cause'], repertoire2['cause']
        effect1, effect2 = repertoire1['effect'], repertoire2['effect']

        # Compute EMD for cause repertoires
        emd_cause = self._compute_emd(cause1, cause2)

        # Compute EMD for effect repertoires
        emd_effect = self._compute_emd(effect1, effect2)

        # Φ is the minimum of the two EMDs (cause and effect)
        phi = min(emd_cause, emd_effect)

        return phi

    def _compute_emd(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Compute Earth Mover's Distance between two probability distributions.
        """
        # For discrete distributions, EMD can be computed using cumulative distributions
        if len(dist1) != len(dist2):
            # Pad shorter distribution
            max_len = max(len(dist1), len(dist2))
            dist1 = np.pad(dist1, (0, max_len - len(dist1)))
            dist2 = np.pad(dist2, (0, max_len - len(dist2)))

        # Compute cumulative distributions
        cdf1 = np.cumsum(dist1)
        cdf2 = np.cumsum(dist2)

        # EMD is the integral of absolute difference of CDFs
        emd = np.sum(np.abs(cdf1 - cdf2)) / len(cdf1)

        return emd

    def compute_consciousness_level(self, system_state: np.ndarray) -> Dict:
        """
        Compute comprehensive consciousness metrics using IIT.
        """
        phi = self.compute_phi(system_state)

        # Additional IIT measures
        repertoire = self.compute_cause_effect_repertoire(system_state)

        # Compute repertoire complexity (diversity)
        cause_entropy = entropy(repertoire['cause'])
        effect_entropy = entropy(repertoire['effect'])

        # Compute integration (how different from independent parts)
        mip = self.find_mip(system_state)
        phi_mip = self.compute_phi(system_state, mip)

        return {
            'phi': phi,  # Integrated information
            'cause_complexity': cause_entropy,
            'effect_complexity': effect_entropy,
            'integration': phi_mip,
            'repertoire_size': len(repertoire['cause']),
            'consciousness_level': self._phi_to_consciousness_level(phi)
        }

    def _phi_to_consciousness_level(self, phi: float) -> str:
        """
        Convert Phi value to qualitative consciousness level.
        """
        if phi < 0.1:
            return "minimal"
        elif phi < 0.5:
            return "basic"
        elif phi < 1.0:
            return "moderate"
        elif phi < 2.0:
            return "advanced"
        else:
            return "high"


class EnhancedGlobalWorkspace:
    """
    Enhanced Global Workspace Theory implementation.
    """
    def __init__(self, num_modules: int = 10):
        self.num_modules = num_modules
        self.modules = {}
        self.workspace_state = None
        self.broadcast_history = []
    
    def register_module(self, module_id: str, module_state: np.ndarray):
        """Register a module with the workspace"""
        self.modules[module_id] = module_state
    
    def compute_competition(self) -> Dict[str, float]:
        """
        Compute competition between modules for workspace access.
        Uses attention-like mechanism.
        """
        scores = {}
        
        for module_id, state in self.modules.items():
            # Score based on activation strength and relevance
            activation = np.linalg.norm(state)
            relevance = self._compute_relevance(module_id)
            score = activation * relevance
            scores[module_id] = score
        
        return scores
    
    def broadcast(self, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Broadcast winning modules to workspace.
        """
        scores = self.compute_competition()
        
        # Select modules above threshold
        winners = {mid: score for mid, score in scores.items() if score >= threshold}
        
        # Combine winner states
        if winners:
            winner_states = [self.modules[mid] for mid in winners.keys()]
            self.workspace_state = np.mean(winner_states, axis=0)
        else:
            # No winners, use average of all
            all_states = list(self.modules.values())
            self.workspace_state = np.mean(all_states, axis=0) if all_states else None
        
        # Record broadcast
        self.broadcast_history.append({
            "winners": list(winners.keys()),
            "scores": scores,
            "timestamp": time.time()
        })

        # Compute synchrony (coherence measure)
        synchrony = np.mean(list(scores.values())) if scores else 0.0

        return self.workspace_state, synchrony
    
    def _compute_relevance(self, module_id: str) -> float:
        """Compute relevance of module (simplified)"""
        # In full implementation, would use context and goals
        return 1.0
    
    def get_workspace_state(self) -> Optional[np.ndarray]:
        """Get current workspace state"""
        return self.workspace_state


class SelfAwareness:
    """
    System that maintains model of itself.
    """
    def __init__(self):
        self.self_model = {}
        self.capabilities = []
        self.limitations = []
        self.goals = []
        self.beliefs = {}
    
    def update_self_model(self, component: str, state: Dict):
        """Update model of a component"""
        self.self_model[component] = state
    
    def reflect_on_capabilities(self) -> Dict:
        """Reflect on own capabilities"""
        return {
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "confidence": self._compute_confidence()
        }
    
    def _compute_confidence(self) -> float:
        """Compute confidence in self-model"""
        # Simplified: based on number of known capabilities
        if self.capabilities:
            return min(1.0, len(self.capabilities) / 10.0)
        return 0.5
    
    def update_beliefs(self, belief: str, confidence: float):
        """Update beliefs about the world"""
        self.beliefs[belief] = confidence
    
    def query_self(self, question: str) -> Dict:
        """Query self-model"""
        if "capability" in question.lower():
            return self.reflect_on_capabilities()
        elif "goal" in question.lower():
            return {"goals": self.goals}
        elif "belief" in question.lower():
            return {"beliefs": self.beliefs}
        else:
            return {"answer": "Unknown query"}


class MetacognitiveMonitoring:
    """
    System that monitors its own cognitive processes.
    """
    def __init__(self):
        self.process_history = []
        self.performance_metrics = {}
        self.error_log = []
    
    def monitor_process(self, process_name: str, start_time: float, end_time: float,
                       success: bool, metadata: Optional[Dict] = None):
        """Monitor a cognitive process"""
        self.process_history.append({
            "process": process_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "success": success,
            "metadata": metadata or {}
        })
        
        # Update performance metrics
        if process_name not in self.performance_metrics:
            self.performance_metrics[process_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "total_duration": 0.0
            }
        
        metrics = self.performance_metrics[process_name]
        metrics["total_calls"] += 1
        if success:
            metrics["successful_calls"] += 1
        metrics["total_duration"] += (end_time - start_time)
    
    def log_error(self, process_name: str, error: Exception, context: Optional[Dict] = None):
        """Log an error"""
        self.error_log.append({
            "process": process_name,
            "error": str(error),
            "context": context or {},
            "timestamp": time.time()
        })
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        report = {}
        for process_name, metrics in self.performance_metrics.items():
            report[process_name] = {
                "success_rate": metrics["successful_calls"] / metrics["total_calls"] if metrics["total_calls"] > 0 else 0,
                "avg_duration": metrics["total_duration"] / metrics["total_calls"] if metrics["total_calls"] > 0 else 0,
                "total_calls": metrics["total_calls"]
            }
        return report
    
    def identify_bottlenecks(self, threshold: float = 1.0) -> List[str]:
        """Identify processes that are bottlenecks"""
        bottlenecks = []
        for process_name, metrics in self.performance_metrics.items():
            avg_duration = metrics["total_duration"] / metrics["total_calls"] if metrics["total_calls"] > 0 else 0
            if avg_duration > threshold:
                bottlenecks.append(process_name)
        return bottlenecks

