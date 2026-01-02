"""
Advanced planning system with HTN, MCTS, and neuro-symbolic reasoning.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn


@dataclass
class Task:
    """Represents a task in HTN planning"""
    name: str
    preconditions: List[str]
    effects: List[str]
    cost: float = 1.0
    primitive: bool = True


@dataclass
class Method:
    """Represents a method for decomposing compound tasks"""
    name: str
    task_name: str
    preconditions: List[str]
    subtasks: List[str]


class HTNPlanner:
    """
    Hierarchical Task Network (HTN) planner.
    """
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.methods: Dict[str, List[Method]] = defaultdict(list)
        self.state: Set[str] = set()

    def add_task(self, task: Task):
        """Add a primitive task"""
        self.tasks[task.name] = task

    def add_method(self, method: Method):
        """Add a decomposition method"""
        self.methods[method.task_name].append(method)

    def set_initial_state(self, state: Set[str]):
        """Set the initial state"""
        self.state = state.copy()

    def plan(self, goal_task: str, max_depth: int = 10) -> Optional[List[str]]:
        """
        Find a plan to achieve the goal task.

        Returns:
            List of primitive task names, or None if no plan found
        """
        plan = []
        if self._decompose_task(goal_task, plan, max_depth):
            return plan
        return None

    def _decompose_task(self, task_name: str, plan: List[str], max_depth: int) -> bool:
        """Recursively decompose tasks"""
        if max_depth <= 0:
            return False

        if task_name in self.tasks and self.tasks[task_name].primitive:
            # Primitive task - add to plan
            plan.append(task_name)
            return True

        # Compound task - try methods
        for method in self.methods[task_name]:
            if self._check_preconditions(method.preconditions):
                # Try this method
                plan_start = len(plan)
                success = True

                for subtask in method.subtasks:
                    if not self._decompose_task(subtask, plan, max_depth - 1):
                        success = False
                        break

                if success:
                    return True
                else:
                    # Backtrack
                    del plan[plan_start:]

        return False

    def _check_preconditions(self, preconditions: List[str]) -> bool:
        """Check if preconditions are satisfied"""
        return all(precond in self.state for precond in preconditions)


class MCTSNode:
    """Node in Monte Carlo Tree Search"""
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action: Optional[Any] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Any] = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        """Select best child using UCB1 formula"""
        if not self.children:
            return None

        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            return child.value / child.visits + c * math.sqrt(math.log(self.visits) / child.visits)

        return max(self.children, key=ucb_score)

    def add_child(self, child: 'MCTSNode') -> 'MCTSNode':
        self.children.append(child)
        return child


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner.
    """
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations

    def plan(self, initial_state: Any, goal_check: Callable[[Any], bool],
             get_actions: Callable[[Any], List[Any]], transition: Callable[[Any, Any], Any],
             get_reward: Callable[[Any], float]) -> List[Any]:
        """
        Plan using MCTS.

        Args:
            initial_state: Starting state
            goal_check: Function to check if state is goal
            get_actions: Function to get possible actions from state
            transition: Function to get next state from state + action
            get_reward: Function to get reward from state

        Returns:
            Best action sequence found
        """
        root = MCTSNode(initial_state)
        root.untried_actions = get_actions(initial_state)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not goal_check(node.state):
                node = self._expand(node, get_actions, transition)
            reward = self._simulate(node, goal_check, get_actions, transition, get_reward)
            self._backpropagate(node, reward)

        # Return best action sequence
        return self._get_best_path(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node using UCB1"""
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def _expand(self, node: MCTSNode, get_actions: Callable, transition: Callable) -> MCTSNode:
        """Expand node by adding new child"""
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        new_state = transition(node.state, action)
        child = MCTSNode(new_state, node, action)
        child.untried_actions = get_actions(new_state)

        return node.add_child(child)

    def _simulate(self, node: MCTSNode, goal_check: Callable, get_actions: Callable,
                 transition: Callable, get_reward: Callable) -> float:
        """Simulate random playout from node"""
        state = node.state
        total_reward = 0.0
        depth = 0
        max_depth = 50

        while not goal_check(state) and depth < max_depth:
            actions = get_actions(state)
            if not actions:
                break

            action = random.choice(actions)
            state = transition(state, action)
            total_reward += get_reward(state)
            depth += 1

        if goal_check(state):
            total_reward += 100.0  # Goal bonus

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _get_best_path(self, root: MCTSNode) -> List[Any]:
        """Get best action sequence from root"""
        path = []
        node = root

        while node.children:
            node = node.best_child()
            if node.action is not None:
                path.append(node.action)

        return path


class NeuroSymbolicReasoner(nn.Module):
    """
    Neuro-symbolic reasoning that combines neural networks with symbolic logic.
    """
    def __init__(self, num_symbols: int, embedding_dim: int = 64):
        super().__init__()
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim

        # Symbol embeddings
        self.symbol_embeddings = nn.Embedding(num_symbols, embedding_dim)

        # Neural rule learner
        self.rule_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),  # premise1 + premise2 + conclusion
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Confidence score
            nn.Sigmoid()
        )

        # Reasoning network
        self.reasoning_net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, premises: torch.Tensor, conclusion: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for rule learning.

        Args:
            premises: [batch, 2, embedding_dim] - two premises
            conclusion: [batch, embedding_dim] - conclusion

        Returns:
            Confidence scores for the rules
        """
        # Combine premises and conclusion
        combined = torch.cat([
            premises[:, 0],  # premise 1
            premises[:, 1],  # premise 2
            conclusion       # conclusion
        ], dim=-1)

        return self.rule_encoder(combined)

    def reason(self, facts: torch.Tensor, rules: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        Perform neuro-symbolic reasoning.

        Args:
            facts: [num_facts, embedding_dim] - known facts
            rules: List of (premise1_idx, premise2_idx, conclusion_idx) tuples

        Returns:
            Derived conclusions
        """
        conclusions = []

        for rule in rules:
            p1_idx, p2_idx, c_idx = rule

            if p1_idx < len(facts) and p2_idx < len(facts):
                premise1 = facts[p1_idx]
                premise2 = facts[p2_idx]

                # Apply rule (simple combination)
                combined = torch.cat([premise1, premise2])
                derived = self.reasoning_net(combined)

                conclusions.append((c_idx, derived))

        return conclusions


class PlanningSystem:
    """
    Complete planning system integrating HTN, MCTS, and neuro-symbolic reasoning.
    """
    def __init__(self, num_symbols: int = 100):
        self.htn_planner = HTNPlanner()
        self.mcts_planner = MCTSPlanner()
        self.neuro_symbolic = NeuroSymbolicReasoner(num_symbols)

        # Initialize with basic tasks and methods
        self._initialize_basic_knowledge()

    def _initialize_basic_knowledge(self):
        """Initialize with basic planning knowledge"""
        # Add some basic primitive tasks
        self.htn_planner.add_task(Task("move", [], ["at_location"]))
        self.htn_planner.add_task(Task("pickup", ["at_location"], ["has_object"]))
        self.htn_planner.add_task(Task("putdown", ["has_object"], ["object_placed"]))

        # Add compound tasks and methods
        method1 = Method(
            "transport_object",
            "transport_object",
            ["at_location"],
            ["move", "pickup", "move", "putdown"]
        )
        self.htn_planner.add_method(method1)

    def plan_with_htn(self, goal: str, initial_state: Set[str]) -> Optional[List[str]]:
        """Plan using HTN"""
        self.htn_planner.set_initial_state(initial_state)
        return self.htn_planner.plan(goal)

    def plan_with_mcts(self, initial_state: Any, goal_check: Callable,
                      get_actions: Callable, transition: Callable,
                      get_reward: Callable) -> List[Any]:
        """Plan using MCTS"""
        return self.mcts_planner.plan(initial_state, goal_check, get_actions,
                                    transition, get_reward)

    def neuro_symbolic_reasoning(self, facts: List[int], rules: List[Tuple[int, int, int]]) -> List[Tuple[int, torch.Tensor]]:
        """
        Perform neuro-symbolic reasoning.

        Args:
            facts: List of symbol indices that are true
            rules: List of (premise1, premise2, conclusion) rules

        Returns:
            Derived conclusions
        """
        # Convert facts to embeddings
        fact_indices = torch.tensor(facts, dtype=torch.long)
        fact_embeddings = self.neuro_symbolic.symbol_embeddings(fact_indices)

        return self.neuro_symbolic.reason(fact_embeddings, rules)

    def integrated_planning(self, goal: str, initial_state: Set[str],
                          facts: List[int] = None, rules: List[Tuple[int, int, int]] = None) -> Dict:
        """
        Integrated planning using all methods.

        Returns:
            Dictionary with planning results from different methods
        """
        results = {}

        # HTN planning
        htn_plan = self.plan_with_htn(goal, initial_state)
        results["htn_plan"] = htn_plan

        # Neuro-symbolic reasoning (if facts provided)
        if facts and rules:
            conclusions = self.neuro_symbolic_reasoning(facts, rules)
            results["neuro_symbolic_conclusions"] = conclusions

        # Combine results
        combined_plan = self._combine_planning_results(results)
        results["combined_plan"] = combined_plan

        return results

    def _combine_planning_results(self, results: Dict) -> List[str]:
        """Combine results from different planning methods"""
        combined = []

        # Use HTN plan as base
        if results.get("htn_plan"):
            combined.extend(results["htn_plan"])

        # Add insights from neuro-symbolic reasoning
        if results.get("neuro_symbolic_conclusions"):
            # This would integrate symbolic insights into the plan
            pass

        return combined

    def learn_from_experience(self, state: Any, action: Any, next_state: Any, reward: float):
        """Learn from planning experience"""
        # This would update the neuro-symbolic reasoner
        # For now, it's a placeholder
        pass
