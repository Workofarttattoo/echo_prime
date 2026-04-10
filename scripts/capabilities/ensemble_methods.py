#!/usr/bin/env python3
"""
ECH0-PRIME Ensemble Methods & Meta-Learning
Implements advanced ensemble reasoning and task adaptation capabilities.
"""

import sys
import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class EnsembleResult:
    """Result from ensemble reasoning."""
    final_answer: str
    confidence_score: float
    method_used: str
    reasoning_steps: List[str]
    consensus_level: float
    alternative_answers: List[Tuple[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningStrategy:
    """A reasoning strategy for ensemble methods."""
    name: str
    domain: str
    temperature: float
    reasoning_depth: int
    prompt_template: str
    confidence_weight: float = 1.0
    success_rate: float = 0.5

class EnsembleReasoner:
    """Advanced ensemble reasoning system with meta-learning capabilities."""
    
    def __init__(self):
        self.strategies = self._load_strategies()
        self.performance_history = self._load_performance_history()
        self.task_adaptation_rules = self._load_adaptation_rules()
        self.confidence_threshold = 0.7
        
    def _load_strategies(self) -> Dict[str, ReasoningStrategy]:
        """Load reasoning strategies from domain configurations."""
        strategies = {}
        
        try:
            from persistent_optimizations import get_persistent_optimization_manager
            pom = get_persistent_optimization_manager()
            
            for domain, config in pom.domain_state.items():
                strategy = ReasoningStrategy(
                    name=f"{domain}_reasoning",
                    domain=domain,
                    temperature=config.get('temperature', 0.3),
                    reasoning_depth=config.get('reasoning_depth', 2),
                    prompt_template=config.get('prompt_template', 'Reason through this: {question}'),
                    confidence_weight=1.0,
                    success_rate=0.5
                )
                strategies[strategy.name] = strategy
            
            # Add specialized ensemble strategies
            strategies['consensus_voting'] = ReasoningStrategy(
                name='consensus_voting',
                domain='general',
                temperature=0.1,
                reasoning_depth=3,
                prompt_template='Provide a well-reasoned answer: {question}',
                confidence_weight=1.2,
                success_rate=0.8
            )
            
            strategies['meta_analysis'] = ReasoningStrategy(
                name='meta_analysis',
                domain='general', 
                temperature=0.2,
                reasoning_depth=4,
                prompt_template='Analyze this problem systematically: {question}',
                confidence_weight=1.1,
                success_rate=0.7
            )
            
        except Exception as e:
            print(f"Warning: Could not load domain strategies: {e}")
            
        return strategies
    
    def _load_performance_history(self) -> Dict[str, List[float]]:
        """Load performance history for meta-learning."""
        try:
            history_file = "optimization_state/ensemble_performance.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load performance history: {e}")
        
        # Default performance history
        return {
            'consensus_voting': [0.8, 0.85, 0.82, 0.87, 0.84],
            'meta_analysis': [0.75, 0.78, 0.82, 0.79, 0.81],
            'domain_specific': [0.7, 0.72, 0.75, 0.73, 0.76]
        }
    
    def _load_adaptation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load task adaptation rules."""
        return {
            'mathematical': {
                'preferred_strategies': ['domain_specific', 'consensus_voting'],
                'confidence_threshold': 0.8,
                'ensemble_size': 3
            },
            'legal': {
                'preferred_strategies': ['meta_analysis', 'domain_specific'],
                'confidence_threshold': 0.85,
                'ensemble_size': 2
            },
            'scientific': {
                'preferred_strategies': ['consensus_voting', 'meta_analysis'],
                'confidence_threshold': 0.75,
                'ensemble_size': 4
            },
            'general': {
                'preferred_strategies': ['consensus_voting', 'meta_analysis', 'domain_specific'],
                'confidence_threshold': 0.7,
                'ensemble_size': 3
            }
        }
    
    async def ensemble_reason(self, question: str, domain: str = 'general', 
                            max_ensemble_size: int = 3) -> EnsembleResult:
        """Perform ensemble reasoning on a question."""
        
        # Determine task type and adaptation rules
        task_type = self._classify_task(question, domain)
        adaptation_rules = self.task_adaptation_rules.get(task_type, self.task_adaptation_rules['general'])
        
        # Select reasoning strategies based on meta-learning
        selected_strategies = self._select_strategies(question, task_type, adaptation_rules, max_ensemble_size)
        
        # Execute ensemble reasoning
        results = await self._execute_ensemble(question, selected_strategies)
        
        # Combine results using consensus methods
        final_result = self._consensus_voting(results, adaptation_rules)
        
        # Update performance history for meta-learning
        self._update_performance_history(final_result, selected_strategies)
        
        return final_result
    
    def _classify_task(self, question: str, domain: str) -> str:
        """Classify the task type for adaptation."""
        question_lower = question.lower()
        
        # Mathematical indicators
        if any(term in question_lower for term in ['solve', 'calculate', 'equation', 'proof', 'theorem']):
            return 'mathematical'
        
        # Legal indicators  
        if any(term in question_lower for term in ['contract', 'breach', 'law', 'legal', 'court', 'statute', 'federal', 'state']):
            return 'legal'
            
        # Scientific indicators
        if any(term in question_lower for term in ['experiment', 'hypothesis', 'theory', 'research', 'data']):
            return 'scientific'
            
        return 'general'
    
    def _select_strategies(self, question: str, task_type: str, 
                          adaptation_rules: Dict[str, Any], max_size: int) -> List[ReasoningStrategy]:
        """Select optimal strategies using meta-learning."""
        
        preferred = adaptation_rules.get('preferred_strategies', ['consensus_voting'])
        available_strategies = [self.strategies[name] for name in preferred if name in self.strategies]
        
        if len(available_strategies) == 0:
            available_strategies = list(self.strategies.values())
        
        # Sort by historical performance and task suitability
        scored_strategies = []
        for strategy in available_strategies:
            # Base score from historical performance
            history = self.performance_history.get(strategy.name, [0.5])
            avg_performance = statistics.mean(history)
            
            # Task suitability bonus
            suitability_bonus = 0.1 if strategy.domain in [task_type, 'general'] else 0.0
            
            # Confidence weight bonus
            confidence_bonus = strategy.confidence_weight * 0.05
            
            total_score = avg_performance + suitability_bonus + confidence_bonus
            scored_strategies.append((strategy, total_score))
        
        # Select top strategies
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        selected = [strategy for strategy, score in scored_strategies[:max_size]]
        
        return selected
    
    async def _execute_ensemble(self, question: str, strategies: List[ReasoningStrategy]) -> List[Dict[str, Any]]:
        """Execute ensemble reasoning with selected strategies."""
        results = []
        
        for strategy in strategies:
            try:
                # Simulate reasoning execution (in real implementation, this would call the actual reasoning engine)
                result = await self._simulate_reasoning(question, strategy)
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Strategy {strategy.name} failed: {e}")
                continue
        
        return results
    
    async def _simulate_reasoning(self, question: str, strategy: ReasoningStrategy) -> Dict[str, Any]:
        """Simulate reasoning execution for ensemble testing."""
        # This would be replaced with actual reasoning engine calls
        
        # Generate pseudo-random but deterministic result based on question and strategy
        import hashlib
        question_hash = hashlib.md5(f"{question}_{strategy.name}".encode()).hexdigest()
        random.seed(int(question_hash[:8], 16))
        
        # Simulate different reasoning approaches
        if 'consensus' in strategy.name:
            answer = f"Consensus analysis: {question[:50]}... requires systematic evaluation."
            confidence = random.uniform(0.75, 0.95)
        elif 'meta' in strategy.name:
            answer = f"Meta-analysis: {question[:50]}... demands structured reasoning approach."  
            confidence = random.uniform(0.70, 0.90)
        else:
            answer = f"Domain-specific analysis: {question[:50]}... using specialized methodology."
            confidence = random.uniform(0.65, 0.85)
        
        reasoning_steps = [
            f"Applied {strategy.name} methodology",
            f"Analyzed problem domain: {strategy.domain}",
            f"Generated reasoned conclusion with confidence {confidence:.2f}"
        ]
        
        return {
            'strategy': strategy.name,
            'answer': answer,
            'confidence': confidence,
            'reasoning_steps': reasoning_steps,
            'metadata': {
                'temperature': strategy.temperature,
                'depth': strategy.reasoning_depth
            }
        }
    
    def _consensus_voting(self, results: List[Dict[str, Any]], 
                         adaptation_rules: Dict[str, Any]) -> EnsembleResult:
        """Combine results using consensus voting."""
        
        if not results:
            return EnsembleResult(
                final_answer="Unable to generate response",
                confidence_score=0.0,
                method_used="fallback",
                reasoning_steps=["No strategies succeeded"],
                consensus_level=0.0,
                alternative_answers=[]
            )
        
        # Group answers by similarity (simplified clustering)
        answer_groups = defaultdict(list)
        for result in results:
            # Simple grouping by first 20 characters
            key = result['answer'][:50].lower().strip()
            answer_groups[key].append(result)
        
        # Find consensus group
        consensus_group = max(answer_groups.values(), key=len)
        consensus_size = len(consensus_group)
        total_results = len(results)
        consensus_level = consensus_size / total_results
        
        # Select best answer from consensus group
        best_result = max(consensus_group, key=lambda x: x['confidence'])
        
        # Calculate weighted confidence
        weights = [r['confidence'] for r in consensus_group]
        weighted_confidence = statistics.mean(weights)
        
        # Prepare alternative answers
        alternatives = []
        for group_key, group in answer_groups.items():
            if group != consensus_group:
                avg_confidence = statistics.mean([r['confidence'] for r in group])
                alternatives.append((group[0]['answer'], avg_confidence))
        
        return EnsembleResult(
            final_answer=best_result['answer'],
            confidence_score=weighted_confidence,
            method_used="consensus_voting",
            reasoning_steps=best_result['reasoning_steps'],
            consensus_level=consensus_level,
            alternative_answers=alternatives,
            metadata={
                'ensemble_size': total_results,
                'consensus_size': consensus_size,
                'strategies_used': [r['strategy'] for r in results]
            }
        )
    
    def _update_performance_history(self, result: EnsembleResult, strategies: List[ReasoningStrategy]):
        """Update performance history for meta-learning."""
        # Record success/failure based on confidence threshold
        success = result.confidence_score >= self.confidence_threshold
        
        for strategy in strategies:
            if strategy.name not in self.performance_history:
                self.performance_history[strategy.name] = []
            
            # Add performance score (simplified)
            performance_score = result.confidence_score if success else result.confidence_score * 0.5
            self.performance_history[strategy.name].append(performance_score)
            
            # Keep only last 10 results
            self.performance_history[strategy.name] = self.performance_history[strategy.name][-10:]
        
        # Save updated history
        try:
            os.makedirs("optimization_state", exist_ok=True)
            with open("optimization_state/ensemble_performance.json", 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save performance history: {e}")

class MetaLearningAdapter:
    """Meta-learning system for task adaptation."""
    
    def __init__(self):
        self.task_patterns = self._load_task_patterns()
        self.adaptation_history = self._load_adaptation_history()
        
    def _load_task_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load learned task patterns."""
        return {
            'complex_reasoning': {
                'indicators': ['therefore', 'because', 'however', 'moreover', 'consequently'],
                'optimal_ensemble_size': 4,
                'preferred_methods': ['meta_analysis', 'consensus_voting']
            },
            'factual_lookup': {
                'indicators': ['what is', 'define', 'explain', 'describe'],
                'optimal_ensemble_size': 2,
                'preferred_methods': ['domain_specific']
            },
            'creative_problem': {
                'indicators': ['design', 'create', 'innovate', 'develop', 'build'],
                'optimal_ensemble_size': 3,
                'preferred_methods': ['consensus_voting', 'meta_analysis']
            }
        }
    
    def _load_adaptation_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load adaptation history."""
        try:
            history_file = "optimization_state/adaptation_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return defaultdict(list)
    
    def adapt_for_task(self, question: str, domain: str) -> Dict[str, Any]:
        """Adapt ensemble parameters based on task analysis."""
        
        # Analyze question characteristics
        question_lower = question.lower()
        word_count = len(question.split())
        complexity_indicators = sum(1 for indicator in ['therefore', 'because', 'however', 'moreover', 'consequently', 'analyze', 'evaluate'] 
                                   if indicator in question_lower)
        
        # Determine task pattern
        task_pattern = 'factual_lookup'
        if complexity_indicators >= 2:
            task_pattern = 'complex_reasoning'
        elif any(word in question_lower for word in ['design', 'create', 'develop']):
            task_pattern = 'creative_problem'
        
        pattern_config = self.task_patterns[task_pattern]
        
        # Adapt based on history
        if domain in self.adaptation_history:
            history = self.adaptation_history[domain]
            if history:
                avg_performance = statistics.mean([h.get('performance', 0.5) for h in history])
                if avg_performance > 0.8:
                    # Increase ensemble size for high-performing domains
                    pattern_config = pattern_config.copy()
                    pattern_config['optimal_ensemble_size'] = min(5, pattern_config['optimal_ensemble_size'] + 1)
        
        return {
            'task_pattern': task_pattern,
            'ensemble_size': pattern_config['optimal_ensemble_size'],
            'preferred_methods': pattern_config['preferred_methods'],
            'confidence_threshold': 0.75 if complexity_indicators >= 2 else 0.65,
            'adaptation_metadata': {
                'word_count': word_count,
                'complexity_score': complexity_indicators,
                'domain': domain
            }
        }
    
    def record_adaptation_result(self, domain: str, adaptation: Dict[str, Any], 
                               performance: float, success: bool):
        """Record adaptation result for learning."""
        result = {
            'timestamp': time.time(),
            'adaptation': adaptation,
            'performance': performance,
            'success': success
        }
        
        self.adaptation_history[domain].append(result)
        
        # Keep only recent results
        self.adaptation_history[domain] = self.adaptation_history[domain][-20:]
        
        # Save history
        try:
            os.makedirs("optimization_state", exist_ok=True)
            with open("optimization_state/adaptation_history.json", 'w') as f:
                json.dump(dict(self.adaptation_history), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save adaptation history: {e}")

# Global instances
_ensemble_reasoner = None
_meta_adapter = None

def get_ensemble_reasoner() -> EnsembleReasoner:
    """Get the global ensemble reasoner."""
    global _ensemble_reasoner
    if _ensemble_reasoner is None:
        _ensemble_reasoner = EnsembleReasoner()
    return _ensemble_reasoner

def get_meta_learning_adapter() -> MetaLearningAdapter:
    """Get the global meta-learning adapter."""
    global _meta_adapter
    if _meta_adapter is None:
        _meta_adapter = MetaLearningAdapter()
    return _meta_adapter
