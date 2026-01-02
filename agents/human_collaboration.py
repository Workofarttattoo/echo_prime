"""
Human-AI collaboration system with interpretable explanations and interactive learning.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class Explanation:
    """Represents an explanation for AI decisions/actions"""
    decision_type: str  # "prediction", "action", "reasoning"
    confidence: float
    evidence: List[str]
    reasoning_chain: List[str]
    alternatives: List[Dict[str, Any]]
    uncertainty_factors: List[str]


@dataclass
class Feedback:
    """Represents human feedback on AI outputs"""
    feedback_type: str  # "correction", "preference", "clarification"
    target_output: Any
    human_input: Any
    context: Dict[str, Any]
    timestamp: float


class ExplanationGenerator:
    """
    Generates interpretable explanations for AI decisions.
    """
    def __init__(self, llm_bridge=None):
        self.llm_bridge = llm_bridge
        self.explanation_templates = {
            "prediction": "I predicted {outcome} because {evidence}. I'm {confidence:.1%} confident.",
            "action": "I chose to {action} because {reasoning}. This leads to {expected_outcome}.",
            "reasoning": "My reasoning follows this chain: {chain}"
        }

    def explain_prediction(self, prediction: Any, features: Dict[str, Any],
                          model_confidence: float) -> Explanation:
        """Generate explanation for a prediction"""
        evidence = self._extract_evidence(features)
        reasoning_chain = self._build_reasoning_chain(prediction, features)

        explanation_text = self.explanation_templates["prediction"].format(
            outcome=str(prediction),
            evidence="; ".join(evidence),
            confidence=model_confidence
        )

        return Explanation(
            decision_type="prediction",
            confidence=model_confidence,
            evidence=evidence,
            reasoning_chain=reasoning_chain,
            alternatives=self._generate_alternatives(prediction),
            uncertainty_factors=self._identify_uncertainties(features)
        )

    def explain_action(self, action: str, state: Dict[str, Any],
                      expected_outcome: str) -> Explanation:
        """Generate explanation for an action"""
        reasoning = self._explain_action_reasoning(action, state)

        explanation_text = self.explanation_templates["action"].format(
            action=action,
            reasoning=reasoning,
            expected_outcome=expected_outcome
        )

        return Explanation(
            decision_type="action",
            confidence=0.8,  # Default confidence for actions
            evidence=[reasoning],
            reasoning_chain=[f"Current state: {state}", f"Chosen action: {action}"],
            alternatives=self._action_alternatives(action, state),
            uncertainty_factors=["External factors", "Incomplete information"]
        )

    def explain_reasoning(self, conclusion: str, premises: List[str]) -> Explanation:
        """Generate explanation for reasoning chain"""
        chain_description = " → ".join(premises + [conclusion])

        explanation_text = self.explanation_templates["reasoning"].format(
            chain=chain_description
        )

        return Explanation(
            decision_type="reasoning",
            confidence=0.9,
            evidence=premises,
            reasoning_chain=premises + [conclusion],
            alternatives=[],
            uncertainty_factors=[]
        )

    def _extract_evidence(self, features: Dict[str, Any]) -> List[str]:
        """Extract key evidence from features"""
        evidence = []
        for key, value in features.items():
            if isinstance(value, (int, float)) and abs(value) > 0.5:
                evidence.append(f"{key} is {value:.2f}")
            elif isinstance(value, bool) and value:
                evidence.append(f"{key} is true")
        return evidence[:5]  # Limit to top 5

    def _build_reasoning_chain(self, prediction: Any, features: Dict[str, Any]) -> List[str]:
        """Build reasoning chain for prediction"""
        return [
            f"Analyzed input features: {list(features.keys())}",
            f"Applied learned patterns to data",
            f"Generated prediction: {prediction}"
        ]

    def _generate_alternatives(self, prediction: Any) -> List[Dict[str, Any]]:
        """Generate alternative predictions"""
        if isinstance(prediction, str):
            return [
                {"alternative": "alternative_1", "confidence": 0.3},
                {"alternative": "alternative_2", "confidence": 0.2}
            ]
        elif isinstance(prediction, (int, float)):
            return [
                {"alternative": prediction * 0.8, "confidence": 0.3},
                {"alternative": prediction * 1.2, "confidence": 0.2}
            ]
        return []

    def _identify_uncertainties(self, features: Dict[str, Any]) -> List[str]:
        """Identify sources of uncertainty"""
        uncertainties = []
        for key, value in features.items():
            if isinstance(value, (int, float)) and abs(value) < 0.3:
                uncertainties.append(f"Low confidence in {key}")
        return uncertainties

    def _explain_action_reasoning(self, action: str, state: Dict[str, Any]) -> str:
        """Explain why an action was chosen"""
        return f"Based on current state analysis, {action} optimizes the next step"

    def _action_alternatives(self, action: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative actions"""
        return [
            {"alternative": f"alternative_to_{action}", "confidence": 0.4},
            {"alternative": "wait_and_observe", "confidence": 0.3}
        ]


class InteractiveLearner:
    """
    Learns from human feedback and corrections.
    """
    def __init__(self, learning_system=None):
        self.learning_system = learning_system
        self.feedback_history = []
        self.correction_patterns = defaultdict(int)

    def process_feedback(self, feedback: Feedback) -> Dict[str, Any]:
        """
        Process human feedback and update learning.

        Returns:
            Learning update results
        """
        self.feedback_history.append(feedback)

        if feedback.feedback_type == "correction":
            return self._learn_from_correction(feedback)
        elif feedback.feedback_type == "preference":
            return self._learn_from_preference(feedback)
        elif feedback.feedback_type == "clarification":
            return self._learn_from_clarification(feedback)

        return {"status": "processed", "feedback_type": feedback.feedback_type}

    def _learn_from_correction(self, feedback: Feedback) -> Dict[str, Any]:
        """Learn from human correction"""
        # Update correction patterns
        correction_key = f"{feedback.target_output} -> {feedback.human_input}"
        self.correction_patterns[correction_key] += 1

        # Update learning system if available
        if self.learning_system:
            # Create loss from correction
            error = torch.tensor(1.0, requires_grad=True)  # Placeholder
            self.learning_system.step(loss=error)

        return {
            "correction_learned": correction_key,
            "pattern_count": self.correction_patterns[correction_key]
        }

    def _learn_from_preference(self, feedback: Feedback) -> Dict[str, Any]:
        """Learn from human preference"""
        # This would update reward functions or preference models
        return {
            "preference_learned": f"Preferred {feedback.human_input} over {feedback.target_output}"
        }

    def _learn_from_clarification(self, feedback: Feedback) -> Dict[str, Any]:
        """Learn from human clarification"""
        # This would update understanding of ambiguous situations
        return {
            "clarification_stored": str(feedback.human_input)
        }

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning history"""
        total_feedback = len(self.feedback_history)
        correction_rate = sum(1 for f in self.feedback_history
                            if f.feedback_type == "correction") / max(total_feedback, 1)

        return {
            "total_feedback": total_feedback,
            "correction_rate": correction_rate,
            "top_corrections": dict(sorted(self.correction_patterns.items(),
                                         key=lambda x: x[1], reverse=True)[:5])
        }


class SharedMentalModel:
    """
    Maintains shared understanding between human and AI.
    """
    def __init__(self):
        self.shared_concepts = {}
        self.alignment_score = 0.0
        self.communication_history = []

    def update_concept(self, concept: str, human_understanding: str,
                      ai_understanding: str) -> float:
        """
        Update shared understanding of a concept.

        Returns:
            Alignment score (0-1)
        """
        # Simplified alignment calculation
        # In practice, would use semantic similarity
        alignment = 1.0 if human_understanding == ai_understanding else 0.5

        self.shared_concepts[concept] = {
            "human": human_understanding,
            "ai": ai_understanding,
            "alignment": alignment,
            "last_updated": torch.tensor(0.0).item()  # timestamp
        }

        self._update_overall_alignment()
        return alignment

    def check_alignment(self, concept: str) -> Dict[str, Any]:
        """Check alignment for a concept"""
        if concept in self.shared_concepts:
            return self.shared_concepts[concept]
        return {"status": "unknown_concept"}

    def get_misaligned_concepts(self, threshold: float = 0.7) -> List[str]:
        """Get concepts with low alignment"""
        misaligned = []
        for concept, data in self.shared_concepts.items():
            if data["alignment"] < threshold:
                misaligned.append(concept)
        return misaligned

    def _update_overall_alignment(self):
        """Update overall alignment score"""
        if not self.shared_concepts:
            self.alignment_score = 0.0
            return

        alignments = [data["alignment"] for data in self.shared_concepts.values()]
        self.alignment_score = sum(alignments) / len(alignments)

    def record_communication(self, speaker: str, message: str, context: Dict[str, Any]):
        """Record communication for analysis"""
        self.communication_history.append({
            "speaker": speaker,
            "message": message,
            "context": context,
            "timestamp": torch.tensor(0.0).item()
        })


class DelegationProtocol:
    """
    Handles task delegation between human and AI.
    """
    def __init__(self):
        self.capability_assessment = {}
        self.delegation_history = []
        self.human_expertise = defaultdict(float)
        self.ai_expertise = defaultdict(float)

    def assess_capabilities(self, task_type: str, human_capability: float,
                           ai_capability: float):
        """Assess capabilities for task delegation"""
        self.capability_assessment[task_type] = {
            "human": human_capability,
            "ai": ai_capability,
            "delegation_preference": "human" if human_capability > ai_capability else "ai"
        }

    def delegate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to delegate task to human or AI.

        Returns:
            Delegation decision
        """
        task_type = task.get("type", "general")
        urgency = task.get("urgency", 0.5)
        complexity = task.get("complexity", 0.5)

        if task_type in self.capability_assessment:
            assessment = self.capability_assessment[task_type]
            human_cap = assessment["human"]
            ai_cap = assessment["ai"]

            # Consider urgency and complexity
            if urgency > 0.8:
                # High urgency: delegate to whoever is faster
                delegate_to = "ai" if ai_cap > human_cap else "human"
            elif complexity > 0.7:
                # High complexity: delegate to more capable
                delegate_to = "human" if human_cap > ai_cap else "ai"
            else:
                # Normal case: use preference
                delegate_to = assessment["delegation_preference"]
        else:
            # Default: delegate to AI
            delegate_to = "ai"

        delegation_record = {
            "task": task,
            "delegated_to": delegate_to,
            "reasoning": f"Based on {task_type} capabilities and task characteristics",
            "timestamp": torch.tensor(0.0).item()
        }

        self.delegation_history.append(delegation_record)

        return {
            "decision": delegate_to,
            "confidence": 0.8,
            "reasoning": delegation_record["reasoning"]
        }

    def update_expertise(self, agent: str, task_type: str, performance: float):
        """Update expertise based on performance"""
        if agent == "human":
            self.human_expertise[task_type] = 0.9 * self.human_expertise[task_type] + 0.1 * performance
        else:
            self.ai_expertise[task_type] = 0.9 * self.ai_expertise[task_type] + 0.1 * performance

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation statistics"""
        total_delegations = len(self.delegation_history)
        human_delegations = sum(1 for d in self.delegation_history if d["delegated_to"] == "human")

        return {
            "total_delegations": total_delegations,
            "human_delegations": human_delegations,
            "ai_delegations": total_delegations - human_delegations,
            "human_percentage": human_delegations / max(total_delegations, 1)
        }


class HumanAICollaborationSystem:
    """
    Complete human-AI collaboration system.
    """
    def __init__(self, llm_bridge=None, learning_system=None):
        self.explanation_generator = ExplanationGenerator(llm_bridge)
        self.interactive_learner = InteractiveLearner(learning_system)
        self.shared_mental_model = SharedMentalModel()
        self.delegation_protocol = DelegationProtocol()

    def explain_decision(self, decision_type: str, **kwargs) -> Explanation:
        """Generate explanation for a decision"""
        if decision_type == "prediction":
            return self.explanation_generator.explain_prediction(**kwargs)
        elif decision_type == "action":
            return self.explanation_generator.explain_action(**kwargs)
        elif decision_type == "reasoning":
            return self.explanation_generator.explain_reasoning(**kwargs)
        else:
            raise ValueError(f"Unknown decision type: {decision_type}")

    def process_feedback(self, feedback: Feedback) -> Dict[str, Any]:
        """Process human feedback"""
        return self.interactive_learner.process_feedback(feedback)

    def update_shared_understanding(self, concept: str, human_view: str,
                                  ai_view: str) -> float:
        """Update shared mental model"""
        return self.shared_mental_model.update_concept(concept, human_view, ai_view)

    def delegate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate task based on capabilities"""
        return self.delegation_protocol.delegate_task(task)

    def get_collaboration_status(self) -> Dict[str, Any]:
        """Get comprehensive collaboration status"""
        return {
            "alignment_score": self.shared_mental_model.alignment_score,
            "learning_insights": self.interactive_learner.get_learning_insights(),
            "delegation_stats": self.delegation_protocol.get_delegation_stats(),
            "misaligned_concepts": self.shared_mental_model.get_misaligned_concepts()
        }

    def collaborative_reasoning(self, problem: str, human_input: str = "") -> Dict[str, Any]:
        """
        Perform collaborative reasoning on a problem.
        """
        # This would integrate human input with AI reasoning
        # For now, return a placeholder
        return {
            "problem": problem,
            "human_input": human_input,
            "collaborative_solution": "Integrated human-AI reasoning result",
            "confidence": 0.85
        }
