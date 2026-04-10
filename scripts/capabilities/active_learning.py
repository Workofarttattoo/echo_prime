#!/usr/bin/env python3
"""
ECH0-PRIME Active Learning System
Identifies difficult examples and focuses learning on challenging cases.
"""

import sys
import os
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ActiveLearningSystem:
    """Active learning system that focuses on difficult examples."""
    
    def __init__(self):
        self.difficulty_metrics = self._load_difficulty_metrics()
        self.learning_priorities = self._load_learning_priorities()
        self.confidence_threshold = 0.6
        
    def _load_difficulty_metrics(self) -> Dict[str, List[float]]:
        """Load historical difficulty metrics."""
        try:
            metrics_file = "optimization_state/difficulty_metrics.json"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Default difficulty metrics
        return {
            'mathematical': [0.3, 0.4, 0.35, 0.45, 0.38],
            'legal': [0.25, 0.3, 0.28, 0.32, 0.29],
            'scientific': [0.35, 0.4, 0.38, 0.42, 0.39],
            'general': [0.2, 0.25, 0.22, 0.28, 0.24]
        }
    
    def _load_learning_priorities(self) -> Dict[str, float]:
        """Load learning priorities for different domains."""
        return {
            'mathematical': 0.9,  # High priority - precise reasoning required
            'legal': 0.85,        # High priority - accuracy critical
            'scientific': 0.8,    # High priority - complex analysis
            'general': 0.6        # Medium priority - basic reasoning
        }
    
    def identify_difficult_examples(self, question: str, domain: str, 
                                  confidence_score: float) -> Dict[str, Any]:
        """Identify if an example is difficult and should be prioritized for learning."""
        
        # Calculate difficulty score
        difficulty_score = 1.0 - confidence_score  # Higher confidence = lower difficulty
        
        # Domain-specific difficulty adjustment
        domain_base_difficulty = statistics.mean(self.difficulty_metrics.get(domain, [0.3]))
        adjusted_difficulty = difficulty_score + (domain_base_difficulty * 0.2)
        
        # Length-based difficulty (longer questions tend to be more complex)
        word_count = len(question.split())
        length_multiplier = min(1.5, word_count / 50)  # Cap at 1.5x for very long questions
        
        final_difficulty = adjusted_difficulty * length_multiplier
        
        # Determine if this is a difficult example
        is_difficult = final_difficulty > 0.4  # Threshold for difficult examples
        
        # Calculate learning priority
        priority = self.learning_priorities.get(domain, 0.5) * final_difficulty
        
        return {
            'is_difficult': is_difficult,
            'difficulty_score': final_difficulty,
            'learning_priority': priority,
            'domain': domain,
            'confidence_score': confidence_score,
            'word_count': word_count,
            'recommend_focus': priority > 0.7
        }
    
    def select_focus_examples(self, recent_questions: List[Dict[str, Any]], 
                            max_examples: int = 10) -> List[Dict[str, Any]]:
        """Select examples that should be prioritized for learning."""
        
        focus_candidates = []
        
        for question_data in recent_questions:
            question = question_data.get('question', '')
            domain = question_data.get('domain', 'general')
            confidence = question_data.get('confidence', 0.5)
            
            difficulty_analysis = self.identify_difficult_examples(question, domain, confidence)
            
            if difficulty_analysis['recommend_focus']:
                candidate = {
                    'question': question,
                    'domain': domain,
                    'difficulty_analysis': difficulty_analysis,
                    'original_data': question_data
                }
                focus_candidates.append(candidate)
        
        # Sort by learning priority
        focus_candidates.sort(key=lambda x: x['difficulty_analysis']['learning_priority'], reverse=True)
        
        # Return top candidates
        return focus_candidates[:max_examples]
    
    def generate_focus_training_data(self, focus_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate targeted training data from difficult examples."""
        
        training_examples = []
        
        for example in focus_examples:
            question = example['question']
            domain = example['domain']
            difficulty = example['difficulty_analysis']
            
            # Create multiple training variations for difficult examples
            variations = self._generate_training_variations(question, domain, difficulty)
            training_examples.extend(variations)
        
        return training_examples
    
    def _generate_training_variations(self, question: str, domain: str, 
                                    difficulty: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate training variations for a difficult example."""
        
        variations = []
        
        # Variation 1: Step-by-step reasoning
        if difficulty['difficulty_score'] > 0.5:
            variation1 = {
                "instruction": f"Explain your step-by-step reasoning for: {question}",
                "input": "",
                "output": f"This is a challenging {domain} problem requiring systematic analysis. Break it down into key components and reason through each step carefully.",
                "domain": domain,
                "category": "active_learning_focus",
                "difficulty": "hard",
                "metadata": {
                    "generated_at": time.time(),
                    "source": "active_learning_system",
                    "original_difficulty": difficulty['difficulty_score'],
                    "focus_reason": "high_difficulty_requires_systematic_reasoning"
                }
            }
            variations.append(variation1)
        
        # Variation 2: Alternative approaches
        if difficulty['learning_priority'] > 0.8:
            variation2 = {
                "instruction": f"Consider multiple approaches to solve: {question}",
                "input": "",
                "output": f"This complex {domain} problem benefits from considering alternative solution strategies. Evaluate different methodologies and their relative strengths.",
                "domain": domain,
                "category": "active_learning_focus", 
                "difficulty": "expert",
                "metadata": {
                    "generated_at": time.time(),
                    "source": "active_learning_system",
                    "original_difficulty": difficulty['difficulty_score'],
                    "focus_reason": "high_priority_requires_multiple_approaches"
                }
            }
            variations.append(variation2)
        
        # Variation 3: Domain-specific deep dive
        if domain in ['legal', 'mathematical', 'scientific']:
            variation3 = {
                "instruction": f"Apply advanced {domain} principles to: {question}",
                "input": "",
                "output": f"This requires deep {domain} expertise. Apply advanced principles, consider edge cases, and provide comprehensive analysis with supporting reasoning.",
                "domain": domain,
                "category": "active_learning_focus",
                "difficulty": "expert",
                "metadata": {
                    "generated_at": time.time(),
                    "source": "active_learning_system", 
                    "original_difficulty": difficulty['difficulty_score'],
                    "focus_reason": "domain_expertise_required"
                }
            }
            variations.append(variation3)
        
        return variations
    
    def update_difficulty_metrics(self, domain: str, new_difficulty_score: float):
        """Update difficulty metrics with new observations."""
        if domain not in self.difficulty_metrics:
            self.difficulty_metrics[domain] = []
        
        self.difficulty_metrics[domain].append(new_difficulty_score)
        
        # Keep only recent metrics
        self.difficulty_metrics[domain] = self.difficulty_metrics[domain][-20:]
        
        # Save updated metrics
        try:
            os.makedirs("optimization_state", exist_ok=True)
            with open("optimization_state/difficulty_metrics.json", 'w') as f:
                json.dump(self.difficulty_metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save difficulty metrics: {e}")

class CurriculumLearningManager:
    """Manages curriculum learning progression."""
    
    def __init__(self):
        self.learning_stages = self._define_learning_stages()
        self.current_stage = "foundation"
        
    def _define_learning_stages(self) -> Dict[str, Dict[str, Any]]:
        """Define learning curriculum stages."""
        return {
            "foundation": {
                "difficulty_range": (0.0, 0.3),
                "focus_areas": ["basic_reasoning", "factual_recall"],
                "ensemble_size": 2,
                "description": "Build fundamental reasoning capabilities"
            },
            "intermediate": {
                "difficulty_range": (0.3, 0.6),
                "focus_areas": ["domain_application", "comparative_analysis"],
                "ensemble_size": 3,
                "description": "Apply reasoning to specific domains"
            },
            "advanced": {
                "difficulty_range": (0.6, 0.8),
                "focus_areas": ["complex_reasoning", "multi_domain_integration"],
                "ensemble_size": 4,
                "description": "Handle complex multi-faceted problems"
            },
            "expert": {
                "difficulty_range": (0.8, 1.0),
                "focus_areas": ["novel_problem_solving", "meta_reasoning"],
                "ensemble_size": 5,
                "description": "Tackle novel and meta-level reasoning challenges"
            }
        }
    
    def assess_learning_stage(self, recent_performance: List[float]) -> str:
        """Assess current learning stage based on performance."""
        if not recent_performance:
            return "foundation"
        
        avg_performance = statistics.mean(recent_performance)
        
        if avg_performance >= 0.8:
            return "expert"
        elif avg_performance >= 0.6:
            return "advanced"
        elif avg_performance >= 0.3:
            return "intermediate"
        else:
            return "foundation"
    
    def get_curriculum_recommendations(self, current_stage: str, 
                                     focus_areas: List[str]) -> Dict[str, Any]:
        """Get curriculum-based learning recommendations."""
        
        stage_config = self.learning_stages[current_stage]
        
        recommendations = {
            "stage": current_stage,
            "difficulty_target": stage_config["difficulty_range"],
            "focus_areas": stage_config["focus_areas"],
            "ensemble_size": stage_config["ensemble_size"],
            "next_stage_criteria": self._get_next_stage_criteria(current_stage),
            "training_priorities": self._calculate_training_priorities(focus_areas, stage_config)
        }
        
        return recommendations
    
    def _get_next_stage_criteria(self, current_stage: str) -> str:
        """Get criteria for advancing to next stage."""
        criteria = {
            "foundation": "Achieve 70%+ accuracy on basic reasoning tasks for 10 consecutive evaluations",
            "intermediate": "Maintain 75%+ accuracy on domain-specific tasks for 15 consecutive evaluations", 
            "advanced": "Achieve 80%+ accuracy on complex multi-domain problems for 20 consecutive evaluations",
            "expert": "Consistently perform at 85%+ on novel problem-solving tasks"
        }
        return criteria.get(current_stage, "Maintain high performance standards")
    
    def _calculate_training_priorities(self, focus_areas: List[str], 
                                    stage_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate training priorities based on current stage."""
        
        priorities = {}
        stage_focus = stage_config["focus_areas"]
        
        # Prioritize current stage focus areas
        for area in stage_focus:
            priorities[area] = 0.8
        
        # Include requested focus areas
        for area in focus_areas:
            if area in priorities:
                priorities[area] += 0.2
            else:
                priorities[area] = 0.6
        
        return priorities

# Global instances
_active_learner = None
_curriculum_manager = None

def get_active_learning_system() -> ActiveLearningSystem:
    """Get the global active learning system."""
    global _active_learner
    if _active_learner is None:
        _active_learner = ActiveLearningSystem()
    return _active_learner

def get_curriculum_learning_manager() -> CurriculumLearningManager:
    """Get the global curriculum learning manager."""
    global _curriculum_manager
    if _curriculum_manager is None:
        _curriculum_manager = CurriculumLearningManager()
    return _curriculum_manager
