#!/usr/bin/env python3
"""
ECH0-PRIME Comprehensive Feedback Loop System
Enables continuous learning from user interactions, performance data, and external information.
"""

import time
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import memory_profiler
from pathlib import Path
import hashlib

# Import system components
from memory.manager import MemoryManager
from reasoning.orchestrator import ReasoningOrchestrator
from learning.meta import CSALearningSystem
from core.engine import FreeEnergyEngine, HierarchicalGenerativeModel
from missions.self_modification import SelfModificationSystem
from wisdom_processor import WisdomProcessor


class FeedbackType(Enum):
    """Types of feedback the system can receive"""
    USER_CORRECTION = "user_correction"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_REPORT = "error_report"
    SUCCESS_INDICATOR = "success_indicator"
    USER_PREFERENCE = "user_preference"
    EXTERNAL_DATA = "external_data"
    SELF_EVALUATION = "self_evaluation"
    ENVIRONMENTAL_FEEDBACK = "environmental_feedback"


class FeedbackPriority(Enum):
    """Priority levels for feedback processing"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FeedbackItem:
    """Individual feedback item"""
    id: str
    type: FeedbackType
    priority: FeedbackPriority
    content: Dict[str, Any]
    source: str
    timestamp: float
    processed: bool = False
    processing_result: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'content': self.content,
            'source': self.source,
            'timestamp': self.timestamp,
            'processed': self.processed,
            'processing_result': self.processing_result,
            'confidence_score': self.confidence_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        return cls(
            id=data['id'],
            type=FeedbackType(data['type']),
            priority=FeedbackPriority(data['priority']),
            content=data['content'],
            source=data['source'],
            timestamp=data['timestamp'],
            processed=data.get('processed', False),
            processing_result=data.get('processing_result'),
            confidence_score=data.get('confidence_score', 0.5)
        )


class FeedbackCollector:
    """
    Collects feedback from multiple sources
    """
    def __init__(self):
        self.feedback_queue = asyncio.Queue()
        self.feedback_history = []
        self.collection_sources = {}
        self.running = False

    def register_source(self, source_name: str, collector_func: Callable):
        """Register a feedback collection source"""
        self.collection_sources[source_name] = collector_func

    async def collect_feedback(self, source_name: str, feedback_type: FeedbackType,
                             content: Dict[str, Any], priority: FeedbackPriority = FeedbackPriority.MEDIUM):
        """Collect a feedback item"""
        feedback_id = f"{source_name}_{int(time.time() * 1000)}_{hash(str(content))}"

        feedback_item = FeedbackItem(
            id=feedback_id,
            type=feedback_type,
            priority=priority,
            content=content,
            source=source_name,
            timestamp=time.time()
        )

        await self.feedback_queue.put(feedback_item)
        self.feedback_history.append(feedback_item)

        return feedback_id

    async def get_pending_feedback(self, max_items: int = 10) -> List[FeedbackItem]:
        """Get pending feedback items"""
        items = []
        for _ in range(min(max_items, self.feedback_queue.qsize())):
            try:
                item = self.feedback_queue.get_nowait()
                items.append(item)
            except asyncio.QueueEmpty:
                break
        return items

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback collection statistics"""
        total_feedback = len(self.feedback_history)
        processed_feedback = len([f for f in self.feedback_history if f.processed])

        type_counts = {}
        for feedback in self.feedback_history:
            fb_type = feedback.type.value
            type_counts[fb_type] = type_counts.get(fb_type, 0) + 1

        priority_counts = {}
        for feedback in self.feedback_history:
            pri = feedback.priority.value
            priority_counts[pri] = priority_counts.get(pri, 0) + 1

        return {
            'total_feedback': total_feedback,
            'processed_feedback': processed_feedback,
            'processing_rate': processed_feedback / total_feedback if total_feedback > 0 else 0,
            'type_distribution': type_counts,
            'priority_distribution': priority_counts,
            'avg_confidence': np.mean([f.confidence_score for f in self.feedback_history]) if self.feedback_history else 0
        }


class FeedbackProcessor:
    """
    Processes and analyzes feedback to generate insights and improvements
    """
    def __init__(self, memory_manager: MemoryManager, reasoning_orchestrator: ReasoningOrchestrator):
        self.memory = memory_manager
        self.reasoner = reasoning_orchestrator
        self.processing_rules = {}
        self.insights_generated = []

    def register_processing_rule(self, feedback_type: FeedbackType, processor_func: Callable):
        """Register a processing rule for a feedback type"""
        self.processing_rules[feedback_type] = processor_func

    async def process_feedback(self, feedback_item: FeedbackItem) -> Dict[str, Any]:
        """Process a feedback item and generate insights"""
        try:
            # Use specific processor if available
            if feedback_item.type in self.processing_rules:
                result = await self.processing_rules[feedback_item.type](feedback_item)
            else:
                # Use general processing
                result = await self._general_feedback_processing(feedback_item)

            # Store processing result
            feedback_item.processing_result = result
            feedback_item.processed = True

            # Generate insights
            insights = await self._generate_insights(feedback_item, result)
            self.insights_generated.extend(insights)

            # Update confidence score based on processing
            feedback_item.confidence_score = result.get('confidence', 0.5)

            return {
                'success': True,
                'result': result,
                'insights': insights,
                'confidence': feedback_item.confidence_score
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }

    async def _general_feedback_processing(self, feedback_item: FeedbackItem) -> Dict[str, Any]:
        """General feedback processing using reasoning"""
        # Use LLM to analyze feedback
        analysis_prompt = f"""
Analyze this feedback and extract key insights:

Feedback Type: {feedback_item.type.value}
Source: {feedback_item.source}
Content: {json.dumps(feedback_item.content, indent=2)}

Provide:
1. Key insights or learnings
2. Recommended actions
3. Confidence level (0-1)
4. Impact assessment (low/medium/high/critical)
"""

        try:
            analysis = self.reasoner.llm_bridge.generate(analysis_prompt)

            # Parse analysis (simplified)
            confidence = 0.7  # Default
            impact = "medium"  # Default

            if "high confidence" in analysis.lower():
                confidence = 0.9
            elif "low confidence" in analysis.lower():
                confidence = 0.3

            if "critical" in analysis.lower():
                impact = "critical"
            elif "low" in analysis.lower():
                impact = "low"

            return {
                'analysis': analysis,
                'confidence': confidence,
                'impact': impact,
                'processed_at': time.time()
            }

        except Exception as e:
            return {
                'error': f'LLM analysis failed: {e}',
                'confidence': 0.1,
                'impact': 'unknown'
            }

    async def _generate_insights(self, feedback_item: FeedbackItem, processing_result: Dict) -> List[Dict]:
        """Generate actionable insights from processed feedback"""
        insights = []

        # Extract insights based on feedback type and processing
        if feedback_item.type == FeedbackType.USER_CORRECTION:
            insights.append({
                'type': 'behavioral_adjustment',
                'description': f'Adjust behavior based on user correction: {feedback_item.content.get("correction", "")}',
                'priority': 'high',
                'action_required': True
            })

        elif feedback_item.type == FeedbackType.PERFORMANCE_METRIC:
            metric_name = feedback_item.content.get('metric_name', 'unknown')
            metric_value = feedback_item.content.get('value', 0)
            insights.append({
                'type': 'performance_optimization',
                'description': f'Optimize {metric_name} (current: {metric_value})',
                'priority': 'medium',
                'action_required': metric_value < 0.7  # Action if performance < 70%
            })

        elif feedback_item.type == FeedbackType.ERROR_REPORT:
            error_type = feedback_item.content.get('error_type', 'unknown')
            insights.append({
                'type': 'error_mitigation',
                'description': f'Implement fix for {error_type} errors',
                'priority': 'high',
                'action_required': True
            })

        # Store insights in memory
        for insight in insights:
            await self._store_insight(insight)

        return insights

    async def _store_insight(self, insight: Dict):
        """Store insight in memory system"""
        insight_vector = np.random.randn(1024).astype(np.float32)  # Placeholder embedding

        metadata = {
            'type': 'feedback_insight',
            'insight_type': insight.get('type'),
            'description': insight.get('description'),
            'priority': insight.get('priority'),
            'action_required': insight.get('action_required'),
            'generated_at': time.time()
        }

        self.memory.process_input(insight_vector, metadata)


class LearningAdapter:
    """
    Adapts system behavior based on feedback and insights
    """
    def __init__(self, model: HierarchicalGenerativeModel, learning_system: CSALearningSystem,
                 self_modifier: SelfModificationSystem):
        self.model = model
        self.learning = learning_system
        self.self_modifier = self_modifier
        self.adaptation_history = []
        self.performance_baseline = {}

    async def adapt_from_feedback(self, feedback_item: FeedbackItem, insights: List[Dict]):
        """Adapt system behavior based on feedback and insights"""
        adaptations = []

        for insight in insights:
            if insight.get('action_required', False):
                adaptation = await self._generate_adaptation(feedback_item, insight)
                if adaptation:
                    adaptations.append(adaptation)

        # Apply adaptations
        for adaptation in adaptations:
            success = await self._apply_adaptation(adaptation)
            adaptation['applied'] = success
            adaptation['applied_at'] = time.time()
            self.adaptation_history.append(adaptation)

        return adaptations

    async def _generate_adaptation(self, feedback_item: FeedbackItem, insight: Dict) -> Optional[Dict]:
        """Generate specific adaptation based on insight"""
        adaptation_type = insight.get('type')

        if adaptation_type == 'behavioral_adjustment':
            return {
                'type': 'behavior_modification',
                'description': f'Modify behavior: {insight["description"]}',
                'target_component': 'reasoning_orchestrator',
                'modification': {
                    'add_constraint': feedback_item.content.get('correction')
                }
            }

        elif adaptation_type == 'performance_optimization':
            return {
                'type': 'parameter_tuning',
                'description': f'Optimize parameters for better {insight["description"]}',
                'target_component': 'learning_system',
                'modification': {
                    'adjust_learning_rate': True,
                    'target_metric': feedback_item.content.get('metric_name')
                }
            }

        elif adaptation_type == 'error_mitigation':
            return {
                'type': 'code_improvement',
                'description': f'Fix {insight["description"]}',
                'target_component': 'self_modifier',
                'modification': {
                    'error_type': feedback_item.content.get('error_type'),
                    'fix_strategy': 'add_error_handling'
                }
            }

        return None

    async def _apply_adaptation(self, adaptation: Dict) -> bool:
        """Apply a specific adaptation"""
        try:
            adaptation_type = adaptation['type']

            if adaptation_type == 'parameter_tuning':
                # Adjust learning parameters
                self.learning.adjust_learning_rate(0.001)  # Conservative adjustment
                return True

            elif adaptation_type == 'code_improvement':
                # Use self-modification to improve code
                error_type = adaptation['modification']['error_type']
                improvement_result = await self.self_modifier.propose_improvement(
                    current_code="# Error handling code",
                    performance_metrics={'error_rate': 0.1},
                    improvement_description=f"Add error handling for {error_type}"
                )
                return improvement_result.get('success', False)

            elif adaptation_type == 'behavior_modification':
                # Modify reasoning behavior (placeholder)
                print(f"Adapting behavior: {adaptation['description']}")
                return True

            return False

        except Exception as e:
            print(f"Adaptation application failed: {e}")
            return False

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        total_adaptations = len(self.adaptation_history)
        successful_adaptations = len([a for a in self.adaptation_history if a.get('applied', False)])

        adaptation_types = {}
        for adaptation in self.adaptation_history:
            ad_type = adaptation.get('type', 'unknown')
            adaptation_types[ad_type] = adaptation_types.get(ad_type, 0) + 1

        return {
            'total_adaptations': total_adaptations,
            'successful_adaptations': successful_adaptations,
            'success_rate': successful_adaptations / total_adaptations if total_adaptations > 0 else 0,
            'adaptation_types': adaptation_types
        }


class ContinuousLearningLoop:
    """
    Main feedback loop that continuously learns and adapts
    """
    def __init__(self, memory_manager: MemoryManager, reasoning_orchestrator: ReasoningOrchestrator,
                 model: HierarchicalGenerativeModel, learning_system: CSALearningSystem,
                 self_modifier: SelfModificationSystem):
        self.memory = memory_manager
        self.reasoner = reasoning_orchestrator
        self.model = model
        self.learning = learning_system
        self.self_modifier = self_modifier

        # Initialize components
        self.collector = FeedbackCollector()
        self.processor = FeedbackProcessor(memory_manager, reasoning_orchestrator)
        self.adapter = LearningAdapter(model, learning_system, self_modifier)

        # Learning state
        self.learning_active = False
        self.learning_cycle_count = 0
        self.last_learning_cycle = 0

        # Setup default feedback sources
        self._setup_default_feedback_sources()

    def _setup_default_feedback_sources(self):
        """Setup default feedback collection sources"""

        # System performance feedback
        async def collect_performance_feedback():
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            if cpu_usage > 80:
                await self.collector.collect_feedback(
                    source_name="system_monitor",
                    feedback_type=FeedbackType.PERFORMANCE_METRIC,
                    content={'metric_name': 'cpu_usage', 'value': cpu_usage / 100.0},
                    priority=FeedbackPriority.HIGH
                )

            if memory_usage > 85:
                await self.collector.collect_feedback(
                    source_name="system_monitor",
                    feedback_type=FeedbackType.PERFORMANCE_METRIC,
                    content={'metric_name': 'memory_usage', 'value': memory_usage / 100.0},
                    priority=FeedbackPriority.CRITICAL
                )

        self.collector.register_source("system_performance", collect_performance_feedback)

        # User interaction feedback
        async def collect_user_feedback():
            # This would integrate with user interaction logs
            pass

        self.collector.register_source("user_interaction", collect_user_feedback)

    async def start_learning_loop(self, cycle_interval: float = 60.0):
        """Start the continuous learning loop"""
        self.learning_active = True
        print("🧠 Starting continuous learning feedback loop...")

        while self.learning_active:
            try:
                # Run learning cycle
                await self._run_learning_cycle()

                # Wait for next cycle
                await asyncio.sleep(cycle_interval)

            except Exception as e:
                print(f"Learning cycle error: {e}")
                await asyncio.sleep(cycle_interval)

    async def _run_learning_cycle(self):
        """Execute one complete learning cycle"""
        self.learning_cycle_count += 1
        cycle_start = time.time()

        print(f"🔄 Learning Cycle #{self.learning_cycle_count}")

        # 1. Collect feedback
        feedback_items = await self.collector.get_pending_feedback(max_items=10)

        if not feedback_items:
            print("   No new feedback to process")
            return

        print(f"   Processing {len(feedback_items)} feedback items...")

        # 2. Process feedback
        processed_results = []
        for feedback_item in feedback_items:
            result = await self.processor.process_feedback(feedback_item)
            processed_results.append((feedback_item, result))

        # 3. Generate adaptations
        total_adaptations = 0
        for feedback_item, result in processed_results:
            if result['success'] and 'insights' in result:
                adaptations = await self.adapter.adapt_from_feedback(
                    feedback_item, result['insights']
                )
                total_adaptations += len(adaptations)

        # 4. Update learning system
        await self._update_learning_system(processed_results)

        # 5. Log cycle results
        cycle_duration = time.time() - cycle_start
        print(f"   Completed in {cycle_duration:.2f}s")
        self.last_learning_cycle = time.time()

    async def _update_learning_system(self, processed_results: List[Tuple[FeedbackItem, Dict]]):
        """Update the core learning system with new insights"""
        # Extract learning signals from feedback
        learning_signals = []

        for feedback_item, result in processed_results:
            if result['success']:
                # Create learning signal based on feedback
                signal = {
                    'type': feedback_item.type.value,
                    'confidence': result.get('confidence', 0.5),
                    'impact': result.get('result', {}).get('impact', 'medium'),
                    'content_hash': hashlib.md5(str(feedback_item.content).encode()).hexdigest()
                }
                learning_signals.append(signal)

        if learning_signals:
            # Update meta-learning system
            reward_signal = np.mean([s['confidence'] for s in learning_signals])
            self.learning.step(loss=torch.tensor(0.1), reward=reward_signal)

            print(f"Learning signal processed with reward: {reward_signal:.3f}")
    async def submit_feedback(self, feedback_type: FeedbackType, content: Dict[str, Any],
                            source: str = "external", priority: FeedbackPriority = FeedbackPriority.MEDIUM):
        """Submit feedback to the learning loop"""
        feedback_id = await self.collector.collect_feedback(
            source_name=source,
            feedback_type=feedback_type,
            content=content,
            priority=priority
        )
        return feedback_id

    def stop_learning_loop(self):
        """Stop the continuous learning loop"""
        self.learning_active = False
        print("🛑 Learning loop stopped")

    async def shutdown(self):
        """Gracefully shutdown the learning loop and its tasks."""
        self.learning_active = False
        # If there's a way to cancel the specific loop task, we'd do it here
        print("🧠 Feedback loop shutdown initiated")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        feedback_stats = self.collector.get_feedback_stats()
        adaptation_stats = self.adapter.get_adaptation_stats()

        return {
            'learning_active': self.learning_active,
            'total_cycles': self.learning_cycle_count,
            'last_cycle_time': self.last_learning_cycle,
            'feedback_stats': feedback_stats,
            'adaptation_stats': adaptation_stats,
            'insights_generated': len(self.processor.insights_generated),
            'memory_items': len(self.memory.episodic.storage) if hasattr(self.memory, 'episodic') else 0
        }

    async def force_learning_cycle(self):
        """Force an immediate learning cycle"""
        await self._run_learning_cycle()


# Convenience functions for easy integration
async def create_feedback_loop(memory_manager=None, reasoning_orchestrator=None,
                              model=None, learning_system=None, self_modifier=None) -> ContinuousLearningLoop:
    """Create and initialize a complete feedback loop system"""

    # Initialize components if not provided
    if memory_manager is None:
        memory_manager = MemoryManager()
    if reasoning_orchestrator is None:
        reasoning_orchestrator = ReasoningOrchestrator()
    if model is None:
        model = HierarchicalGenerativeModel()
    if learning_system is None:
        learning_system = CSALearningSystem()
    if self_modifier is None:
        self_modifier = SelfModificationSystem()

    # Create feedback loop
    feedback_loop = ContinuousLearningLoop(
        memory_manager, reasoning_orchestrator, model,
        learning_system, self_modifier
    )

    return feedback_loop


if __name__ == "__main__":
    # Example usage
    async def demo_feedback_loop():
        # Create feedback loop
        feedback_loop = await create_feedback_loop()

        # Submit some example feedback
        await feedback_loop.submit_feedback(
            FeedbackType.PERFORMANCE_METRIC,
            {'metric_name': 'accuracy', 'value': 0.85},
            source="demo"
        )

        await feedback_loop.submit_feedback(
            FeedbackType.USER_CORRECTION,
            {'correction': 'response was too verbose'},
            source="user",
            priority=FeedbackPriority.HIGH
        )

        # Run a learning cycle
        await feedback_loop.force_learning_cycle()

        # Get stats
        stats = feedback_loop.get_learning_stats()
        print("Learning Stats:", json.dumps(stats, indent=2))

    asyncio.run(demo_feedback_loop())
