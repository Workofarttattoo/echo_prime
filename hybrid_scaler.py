#!/usr/bin/env python3
"""
ECH0-PRIME Hybrid Scaling System
Dynamically scales system capabilities for different tasks
"""

import psutil
import torch
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import gc


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ResourceLevel(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


@dataclass
class ScalingConfiguration:
    """Configuration for system scaling"""
    task_type: str
    complexity: TaskComplexity
    resource_level: ResourceLevel
    parallel_workers: int
    memory_limit: int  # MB
    time_limit: int    # seconds
    precision_level: str
    optimization_level: str


@dataclass
class SystemResources:
    """Current system resource status"""
    cpu_percent: float
    memory_percent: float
    memory_used: int  # MB
    gpu_memory_used: Optional[int]  # MB
    gpu_memory_total: Optional[int]  # MB
    active_threads: int


class HybridScaler:
    """
    Hybrid scaling system that adapts system capabilities to task requirements
    """

    def __init__(self):
        self.current_config = None
        self.resource_monitor = ResourceMonitor()
        self.scaling_history = []
        self.performance_profiles = {}

        # Scaling strategies
        self.scaling_strategies = {
            'math_simple': self._scale_for_simple_math,
            'math_complex': self._scale_for_complex_math,
            'vision_basic': self._scale_for_basic_vision,
            'vision_advanced': self._scale_for_advanced_vision,
            'reasoning_light': self._scale_for_light_reasoning,
            'reasoning_heavy': self._scale_for_heavy_reasoning,
            'search_basic': self._scale_for_basic_search,
            'search_exhaustive': self._scale_for_exhaustive_search,
        }

    def scale_to_task(self, task_type: str, complexity: str = 'medium') -> Dict[str, Any]:
        """
        Dynamically scale system for specific task

        Args:
            task_type: Type of task (e.g., 'math', 'vision', 'reasoning')
            complexity: Task complexity ('simple', 'medium', 'complex', 'expert')

        Returns:
            Scaling configuration and recommendations
        """
        # Determine task complexity
        try:
            task_complexity = TaskComplexity(complexity.lower())
        except ValueError:
            task_complexity = TaskComplexity.MODERATE

        # Get current system resources
        current_resources = self.resource_monitor.get_resource_status()

        # Determine optimal resource level
        resource_level = self._determine_resource_level(task_complexity, current_resources)

        # Get task-specific scaling strategy
        scaling_key = f"{task_type}_{complexity.lower()}"
        if scaling_key not in self.scaling_strategies:
            scaling_key = f"{task_type}_moderate"  # fallback

        scaling_strategy = self.scaling_strategies.get(scaling_key, self._scale_default)

        # Generate scaling configuration
        config = scaling_strategy(task_complexity, resource_level, current_resources)

        # Apply scaling configuration
        self._apply_scaling_configuration(config)

        # Store configuration
        self.current_config = config

        scaling_result = {
            'task_type': task_type,
            'complexity': complexity,
            'resource_level': resource_level.value,
            'configuration': self._config_to_dict(config),
            'resource_status': self._resources_to_dict(current_resources),
            'performance_estimates': self._estimate_performance(config),
            'recommendations': self._generate_scaling_recommendations(config, current_resources)
        }

        self.scaling_history.append(scaling_result)

        return scaling_result

    def hybrid_process(self, input_data: Any, task_type: str) -> Dict[str, Any]:
        """
        Process input using hybrid specialized/general approach

        Args:
            input_data: Input data to process
            task_type: Type of task

        Returns:
            Processing results
        """
        # Scale system for the task
        scaling_config = self.scale_to_task(task_type)

        # Determine processing strategy
        if task_type.startswith('math'):
            result = self._process_mathematical(input_data, scaling_config)
        elif task_type.startswith('vision'):
            result = self._process_visual(input_data, scaling_config)
        elif task_type.startswith('reasoning'):
            result = self._process_reasoning(input_data, scaling_config)
        else:
            result = self._process_general(input_data, scaling_config)

        # Add scaling metadata
        result['scaling_info'] = scaling_config

        return result

    def _determine_resource_level(self, complexity: TaskComplexity,
                                resources: SystemResources) -> ResourceLevel:
        """Determine appropriate resource level based on task complexity and available resources"""

        # Base resource level on complexity
        base_level = {
            TaskComplexity.SIMPLE: ResourceLevel.MINIMAL,
            TaskComplexity.MODERATE: ResourceLevel.STANDARD,
            TaskComplexity.COMPLEX: ResourceLevel.ENHANCED,
            TaskComplexity.EXPERT: ResourceLevel.MAXIMUM
        }[complexity]

        # Adjust based on available resources
        if resources.memory_percent > 90 or resources.cpu_percent > 95:
            # System under heavy load - reduce resource usage
            if base_level == ResourceLevel.MAXIMUM:
                return ResourceLevel.ENHANCED
            elif base_level == ResourceLevel.ENHANCED:
                return ResourceLevel.STANDARD
            else:
                return ResourceLevel.MINIMAL

        elif resources.memory_percent < 30 and resources.cpu_percent < 50:
            # System has spare capacity - can use more resources
            if base_level == ResourceLevel.MINIMAL:
                return ResourceLevel.STANDARD
            elif base_level == ResourceLevel.STANDARD:
                return ResourceLevel.ENHANCED

        return base_level

    def _scale_for_simple_math(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                             resources: SystemResources) -> ScalingConfiguration:
        """Scaling for simple mathematical tasks"""
        config = ScalingConfiguration(
            task_type="math_simple",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=1,
            memory_limit=100,  # MB
            time_limit=30,     # seconds
            precision_level="standard",
            optimization_level="basic"
        )

        # Adjust based on resource level
        if resource_level == ResourceLevel.ENHANCED:
            config.parallel_workers = 2
            config.precision_level = "high"
        elif resource_level == ResourceLevel.MAXIMUM:
            config.parallel_workers = 4
            config.precision_level = "maximum"

        return config

    def _scale_for_complex_math(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                              resources: SystemResources) -> ScalingConfiguration:
        """Scaling for complex mathematical tasks"""
        config = ScalingConfiguration(
            task_type="math_complex",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=2,
            memory_limit=500,  # MB
            time_limit=300,    # seconds
            precision_level="high",
            optimization_level="advanced"
        )

        # Adjust based on resource level
        if resource_level == ResourceLevel.MAXIMUM:
            config.parallel_workers = min(8, resources.active_threads + 4)
            config.memory_limit = 2000
            config.precision_level = "maximum"
        elif resource_level == ResourceLevel.STANDARD:
            config.parallel_workers = 1
            config.memory_limit = 200

        return config

    def _scale_for_basic_vision(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                              resources: SystemResources) -> ScalingConfiguration:
        """Scaling for basic vision tasks"""
        config = ScalingConfiguration(
            task_type="vision_basic",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=1,
            memory_limit=200,  # MB
            time_limit=60,     # seconds
            precision_level="standard",
            optimization_level="basic"
        )

        # Vision tasks benefit from GPU if available
        if resources.gpu_memory_total and resources.gpu_memory_total > 1000:
            config.memory_limit = 1000

        return config

    def _scale_for_advanced_vision(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                                 resources: SystemResources) -> ScalingConfiguration:
        """Scaling for advanced vision tasks"""
        config = ScalingConfiguration(
            task_type="vision_advanced",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=2,
            memory_limit=1000,  # MB
            time_limit=300,     # seconds
            precision_level="high",
            optimization_level="advanced"
        )

        # Advanced vision needs significant resources
        if resource_level == ResourceLevel.MAXIMUM:
            config.parallel_workers = 4
            config.memory_limit = 4000

        return config

    def _scale_for_light_reasoning(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                                 resources: SystemResources) -> ScalingConfiguration:
        """Scaling for light reasoning tasks"""
        config = ScalingConfiguration(
            task_type="reasoning_light",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=1,
            memory_limit=100,  # MB
            time_limit=60,     # seconds
            precision_level="standard",
            optimization_level="basic"
        )

        return config

    def _scale_for_heavy_reasoning(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                                 resources: SystemResources) -> ScalingConfiguration:
        """Scaling for heavy reasoning tasks"""
        config = ScalingConfiguration(
            task_type="reasoning_heavy",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=4,
            memory_limit=2000,  # MB
            time_limit=600,     # seconds
            precision_level="high",
            optimization_level="maximum"
        )

        return config

    def _scale_for_basic_search(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                              resources: SystemResources) -> ScalingConfiguration:
        """Scaling for basic search tasks"""
        config = ScalingConfiguration(
            task_type="search_basic",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=2,
            memory_limit=300,  # MB
            time_limit=120,    # seconds
            precision_level="standard",
            optimization_level="basic"
        )

        return config

    def _scale_for_exhaustive_search(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                                   resources: SystemResources) -> ScalingConfiguration:
        """Scaling for exhaustive search tasks"""
        config = ScalingConfiguration(
            task_type="search_exhaustive",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=8,
            memory_limit=4000,  # MB
            time_limit=3600,    # seconds
            precision_level="maximum",
            optimization_level="maximum"
        )

        return config

    def _scale_default(self, complexity: TaskComplexity, resource_level: ResourceLevel,
                     resources: SystemResources) -> ScalingConfiguration:
        """Default scaling configuration"""
        return ScalingConfiguration(
            task_type="general",
            complexity=complexity,
            resource_level=resource_level,
            parallel_workers=2,
            memory_limit=500,
            time_limit=180,
            precision_level="standard",
            optimization_level="basic"
        )

    def _apply_scaling_configuration(self, config: ScalingConfiguration):
        """Apply the scaling configuration to the system"""
        # Memory limit setting disabled for compatibility
        # In production, this would be handled by the deployment environment
        pass

        # Configure thread pool size
        # This is a simplified implementation - in practice would configure actual thread pools

        # Set precision levels
        if config.precision_level == "maximum":
            # Enable high precision mode
            torch.set_default_dtype(torch.float64)
        elif config.precision_level == "high":
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float16)

        # Configure optimization
        if config.optimization_level == "maximum":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        elif config.optimization_level == "advanced":
            torch.backends.cudnn.benchmark = True

    def _process_mathematical(self, input_data: Any, scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process mathematical input with scaled configuration"""
        # This would integrate with the mathematical verification system
        return {
            'result_type': 'mathematical',
            'input': str(input_data),
            'confidence': 0.8,
            'method': 'scaled_mathematical_processing'
        }

    def _process_visual(self, input_data: Any, scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process visual input with scaled configuration"""
        # This would integrate with the vision system
        return {
            'result_type': 'visual',
            'input': str(input_data),
            'confidence': 0.7,
            'method': 'scaled_visual_processing'
        }

    def _process_reasoning(self, input_data: Any, scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process reasoning input with scaled configuration"""
        # This would integrate with the reasoning system
        return {
            'result_type': 'reasoning',
            'input': str(input_data),
            'confidence': 0.75,
            'method': 'scaled_reasoning_processing'
        }

    def _process_general(self, input_data: Any, scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process general input with scaled configuration"""
        return {
            'result_type': 'general',
            'input': str(input_data),
            'confidence': 0.6,
            'method': 'scaled_general_processing'
        }

    def _config_to_dict(self, config: ScalingConfiguration) -> Dict[str, Any]:
        """Convert scaling configuration to dictionary"""
        return {
            'task_type': config.task_type,
            'complexity': config.complexity.value,
            'resource_level': config.resource_level.value,
            'parallel_workers': config.parallel_workers,
            'memory_limit': config.memory_limit,
            'time_limit': config.time_limit,
            'precision_level': config.precision_level,
            'optimization_level': config.optimization_level
        }

    def _resources_to_dict(self, resources: SystemResources) -> Dict[str, Any]:
        """Convert system resources to dictionary"""
        return {
            'cpu_percent': resources.cpu_percent,
            'memory_percent': resources.memory_percent,
            'memory_used': resources.memory_used,
            'gpu_memory_used': resources.gpu_memory_used,
            'gpu_memory_total': resources.gpu_memory_total,
            'active_threads': resources.active_threads
        }

    def _estimate_performance(self, config: ScalingConfiguration) -> Dict[str, Any]:
        """Estimate performance for given configuration"""
        # Base performance estimates
        base_performance = {
            'estimated_speedup': config.parallel_workers,
            'memory_efficiency': 1.0 - (config.memory_limit / 8000),  # Assuming 8GB max
            'expected_accuracy': 0.8 + (config.resource_level.value == 'maximum') * 0.15,
            'processing_time_estimate': config.time_limit * 0.8  # Conservative estimate
        }

        # Adjust based on optimization level
        if config.optimization_level == "maximum":
            base_performance['estimated_speedup'] *= 1.5
        elif config.optimization_level == "advanced":
            base_performance['estimated_speedup'] *= 1.2

        return base_performance

    def _generate_scaling_recommendations(self, config: ScalingConfiguration,
                                        resources: SystemResources) -> List[str]:
        """Generate recommendations for scaling configuration"""
        recommendations = []

        # Resource-based recommendations
        if resources.memory_percent > 80:
            recommendations.append("High memory usage detected - consider reducing parallel workers")
        elif resources.memory_percent < 20:
            recommendations.append("Available memory - could increase resource allocation")

        if resources.cpu_percent > 90:
            recommendations.append("High CPU usage - consider task prioritization")
        elif resources.cpu_percent < 30:
            recommendations.append("Low CPU usage - could parallelize more operations")

        # Configuration-based recommendations
        if config.precision_level == "maximum" and config.complexity == TaskComplexity.SIMPLE:
            recommendations.append("Maximum precision may be overkill for simple tasks")

        if config.parallel_workers > 4 and config.complexity == TaskComplexity.SIMPLE:
            recommendations.append("High parallelization may not be necessary for simple tasks")

        # Task-specific recommendations
        if config.task_type.startswith("vision") and not resources.gpu_memory_total:
            recommendations.append("GPU acceleration recommended for vision tasks")

        return recommendations

    def get_scaling_analytics(self) -> Dict[str, Any]:
        """Get analytics on scaling performance"""
        if not self.scaling_history:
            return {'message': 'No scaling history available'}

        # Analyze scaling patterns
        task_types = [h['task_type'] for h in self.scaling_history]
        complexities = [h['complexity'] for h in self.scaling_history]
        resource_levels = [h['resource_level'] for h in self.scaling_history]

        analytics = {
            'total_scalings': len(self.scaling_history),
            'most_common_task': max(set(task_types), key=task_types.count) if task_types else None,
            'most_common_complexity': max(set(complexities), key=complexities.count) if complexities else None,
            'resource_level_distribution': {
                level: resource_levels.count(level) for level in set(resource_levels)
            },
            'scaling_efficiency': self._calculate_scaling_efficiency(),
            'recommendations': self._generate_analytics_recommendations()
        }

        return analytics

    def _calculate_scaling_efficiency(self) -> float:
        """Calculate overall scaling efficiency"""
        if len(self.scaling_history) < 2:
            return 0.5

        # Simple efficiency metric based on resource utilization
        efficiencies = []
        for history_item in self.scaling_history[-10:]:  # Last 10 scalings
            resource_status = history_item.get('resource_status', {})
            memory_usage = resource_status.get('memory_percent', 50)
            cpu_usage = resource_status.get('cpu_percent', 50)

            # Efficiency is optimal when resources are well-utilized but not overloaded
            memory_efficiency = 1.0 - abs(memory_usage - 70) / 70  # Optimal at 70%
            cpu_efficiency = 1.0 - abs(cpu_usage - 70) / 70

            efficiencies.append((memory_efficiency + cpu_efficiency) / 2)

        return np.mean(efficiencies) if efficiencies else 0.5

    def _generate_analytics_recommendations(self) -> List[str]:
        """Generate recommendations based on scaling analytics"""
        recommendations = []

        analytics = self.get_scaling_analytics()

        efficiency = analytics.get('scaling_efficiency', 0.5)

        if efficiency < 0.3:
            recommendations.append("Low scaling efficiency - review resource allocation strategies")
        elif efficiency > 0.8:
            recommendations.append("Good scaling efficiency - current strategies are effective")

        # Task-specific recommendations
        most_common_task = analytics.get('most_common_task')
        if most_common_task:
            recommendations.append(f"Optimize scaling for frequent task type: {most_common_task}")

        return recommendations


class ResourceMonitor:
    """
    Monitors system resources for scaling decisions
    """

    def __init__(self):
        self.monitoring_active = False
        self.resource_history = []

    def get_resource_status(self) -> SystemResources:
        """Get current system resource status"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used // (1024 * 1024)  # Convert to MB

        # GPU memory (if available)
        gpu_memory_used = None
        gpu_memory_total = None

        try:
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() // (1024 * 1024)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except:
            pass

        active_threads = threading.active_count()

        resources = SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            active_threads=active_threads
        )

        # Store in history
        self.resource_history.append({
            'timestamp': time.time(),
            'resources': self._resources_to_dict(resources)
        })

        # Keep only last 100 entries
        if len(self.resource_history) > 100:
            self.resource_history = self.resource_history[-100:]

        return resources

    def _resources_to_dict(self, resources: SystemResources) -> Dict[str, Any]:
        """Convert resources to dictionary"""
        return {
            'cpu_percent': resources.cpu_percent,
            'memory_percent': resources.memory_percent,
            'memory_used': resources.memory_used,
            'gpu_memory_used': resources.gpu_memory_used,
            'gpu_memory_total': resources.gpu_memory_total,
            'active_threads': resources.active_threads
        }

    def get_resource_trends(self) -> Dict[str, Any]:
        """Get resource usage trends"""
        if len(self.resource_history) < 2:
            return {'message': 'Insufficient data for trend analysis'}

        recent_history = self.resource_history[-20:]  # Last 20 measurements

        cpu_trend = self._calculate_trend([h['resources']['cpu_percent'] for h in recent_history])
        memory_trend = self._calculate_trend([h['resources']['memory_percent'] for h in recent_history])

        return {
            'cpu_trend': cpu_trend,  # Percentage points per measurement
            'memory_trend': memory_trend,
            'cpu_avg': np.mean([h['resources']['cpu_percent'] for h in recent_history]),
            'memory_avg': np.mean([h['resources']['memory_percent'] for h in recent_history]),
            'peak_cpu': max([h['resources']['cpu_percent'] for h in recent_history]),
            'peak_memory': max([h['resources']['memory_percent'] for h in recent_history])
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        return slope
