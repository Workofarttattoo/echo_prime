#!/usr/bin/env python3
"""
ECH0-PRIME Advanced Performance Profiling System
Real-time performance monitoring, bottleneck identification, and optimization recommendations
"""

import time
import psutil
import threading
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import gc
import sys
import os
import inspect
import cProfile
import pstats
import io
from functools import wraps
import asyncio
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("PerformanceProfiler")


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: int  # MB
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None
    active_threads: int = 0
    open_files: int = 0
    network_connections: int = 0
    disk_io_read: int = 0
    disk_io_write: int = 0
    context_switches: int = 0
    page_faults: int = 0


@dataclass
class FunctionProfile:
    """Function-level performance profile"""
    function_name: str
    module_name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    max_time: float = 0.0
    min_time: float = float('inf')
    memory_delta: int = 0
    last_called: Optional[datetime] = None
    bottlenecks: List[str] = field(default_factory=list)


@dataclass
class BottleneckAnalysis:
    """Analysis of system bottlenecks"""
    bottleneck_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    impact_score: float
    recommendations: List[str]
    estimated_fix_time: str
    confidence: float


@dataclass
class OptimizationRecommendation:
    """Specific optimization recommendation"""
    optimization_type: str
    target_component: str
    description: str
    expected_improvement: str
    implementation_complexity: str
    priority: str
    code_changes: List[str]
    risk_assessment: str


class PerformanceProfiler:
    """
    Advanced performance profiling system with real-time monitoring and optimization
    """

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.function_profiles = {}
        self.bottleneck_history = []
        self.optimization_recommendations = []

        # Profiling state
        self.profiler = cProfile.Profile()
        self.profiling_active = False

        # Baseline metrics
        self.baseline_metrics = None
        self.baseline_captured = False

        # Performance thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 80, 'critical': 95},
            'gpu_memory_percent': {'warning': 85, 'critical': 95},
            'response_time': {'warning': 2.0, 'critical': 5.0},
            'memory_leak_rate': {'warning': 10, 'critical': 50}  # MB per minute
        }

        # Performance prediction model
        self.performance_model = None
        self.prediction_history = []

    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("🔍 Advanced Performance Monitoring Started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("🛑 Performance Monitoring Stopped")

    def capture_baseline(self):
        """Capture baseline performance metrics"""
        print("📊 Capturing Performance Baseline...")
        baseline_samples = []

        # Collect samples for 30 seconds
        for _ in range(30):
            metrics = self._collect_metrics()
            baseline_samples.append(metrics)
            time.sleep(1.0)

        # Calculate baseline averages
        self.baseline_metrics = self._average_metrics(baseline_samples)
        self.baseline_captured = True
        print("✅ Baseline Performance Captured")

        return self.baseline_metrics

    def start_profiling(self):
        """Start function-level profiling"""
        if not self.profiling_active:
            self.profiler.enable()
            self.profiling_active = True
            print("⚡ Function-Level Profiling Started")

    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return analysis"""
        if not self.profiling_active:
            return {}

        self.profiler.disable()
        self.profiling_active = False

        # Analyze profiling results
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()

        analysis = self._analyze_profiling_results(s.getvalue())
        print("⚡ Function-Level Profiling Completed")

        return analysis

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile individual functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory

                self._record_function_profile(func, execution_time, memory_delta)

        return wrapper

    def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze current system bottlenecks"""
        if not self.metrics_history:
            return []

        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        bottlenecks = []

        # CPU bottleneck analysis
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        if avg_cpu > self.thresholds['cpu_percent']['critical']:
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='cpu_overload',
                severity='critical',
                description=f'CPU utilization at {avg_cpu:.1f}%, exceeding critical threshold',
                impact_score=0.9,
                recommendations=[
                    'Reduce parallel processing threads',
                    'Optimize CPU-intensive operations',
                    'Consider task offloading to GPU',
                    'Implement CPU usage limits'
                ],
                estimated_fix_time='2-4 hours',
                confidence=0.95
            ))
        elif avg_cpu > self.thresholds['cpu_percent']['warning']:
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='cpu_high_usage',
                severity='high',
                description=f'CPU utilization at {avg_cpu:.1f}%, exceeding warning threshold',
                impact_score=0.7,
                recommendations=[
                    'Monitor CPU usage trends',
                    'Optimize computational kernels',
                    'Consider background task scheduling'
                ],
                estimated_fix_time='1-2 hours',
                confidence=0.85
            ))

        # Memory bottleneck analysis
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        if avg_memory > self.thresholds['memory_percent']['critical']:
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='memory_overload',
                severity='critical',
                description=f'Memory utilization at {avg_memory:.1f}%, exceeding critical threshold',
                impact_score=0.95,
                recommendations=[
                    'Implement memory cleanup routines',
                    'Reduce batch sizes in data processing',
                    'Enable garbage collection optimization',
                    'Consider memory-mapped files for large datasets'
                ],
                estimated_fix_time='3-6 hours',
                confidence=0.9
            ))
        elif avg_memory > self.thresholds['memory_percent']['warning']:
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='memory_high_usage',
                severity='high',
                description=f'Memory utilization at {avg_memory:.1f}%, exceeding warning threshold',
                impact_score=0.6,
                recommendations=[
                    'Monitor memory allocation patterns',
                    'Implement object pooling for frequently used objects',
                    'Check for memory leaks in long-running processes'
                ],
                estimated_fix_time='1-3 hours',
                confidence=0.8
            ))

        # GPU memory analysis
        gpu_metrics = [m for m in recent_metrics if m.gpu_memory_used is not None]
        if gpu_metrics:
            avg_gpu_memory = np.mean([
                (m.gpu_memory_used / m.gpu_memory_total) * 100
                for m in gpu_metrics if m.gpu_memory_total and m.gpu_memory_total > 0
            ])

            if avg_gpu_memory > self.thresholds['gpu_memory_percent']['critical']:
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type='gpu_memory_overload',
                    severity='critical',
                    description=f'GPU memory utilization at {avg_gpu_memory:.1f}%, exceeding critical threshold',
                    impact_score=0.85,
                    recommendations=[
                        'Reduce model size or batch size',
                        'Implement gradient checkpointing',
                        'Use mixed precision training',
                        'Clear GPU cache periodically'
                    ],
                    estimated_fix_time='2-5 hours',
                    confidence=0.9
                ))

        # Threading bottlenecks
        avg_threads = np.mean([m.active_threads for m in recent_metrics])
        if avg_threads > 50:  # Arbitrary threshold
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='threading_overload',
                severity='medium',
                description=f'High thread count: {avg_threads:.0f} active threads',
                impact_score=0.5,
                recommendations=[
                    'Implement thread pooling',
                    'Use async/await patterns instead of threads',
                    'Consolidate concurrent operations',
                    'Monitor thread lifecycle'
                ],
                estimated_fix_time='4-8 hours',
                confidence=0.75
            ))

        # Function-level bottlenecks
        function_bottlenecks = self._analyze_function_bottlenecks()
        bottlenecks.extend(function_bottlenecks)

        self.bottleneck_history.extend(bottlenecks)
        return bottlenecks

    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate automated optimization recommendations"""
        recommendations = []

        bottlenecks = self.analyze_bottlenecks()
        metrics_trends = self._analyze_metrics_trends()

        # CPU optimization recommendations
        if any(b.bottleneck_type.startswith('cpu') for b in bottlenecks):
            recommendations.append(OptimizationRecommendation(
                optimization_type='cpu_optimization',
                target_component='compute_kernels',
                description='Optimize CPU-intensive operations using vectorization and parallel processing',
                expected_improvement='30-50% reduction in CPU usage',
                implementation_complexity='medium',
                priority='high',
                code_changes=[
                    'Use numpy vectorized operations instead of loops',
                    'Implement multiprocessing for CPU-bound tasks',
                    'Profile and optimize hot code paths'
                ],
                risk_assessment='Low risk, performance improvement focused'
            ))

        # Memory optimization recommendations
        if any(b.bottleneck_type.startswith('memory') for b in bottlenecks):
            recommendations.append(OptimizationRecommendation(
                optimization_type='memory_optimization',
                target_component='memory_management',
                description='Implement advanced memory management with pooling and cleanup',
                expected_improvement='40-60% reduction in memory usage',
                implementation_complexity='medium',
                priority='high',
                code_changes=[
                    'Implement object pooling for expensive objects',
                    'Use memory-mapped files for large datasets',
                    'Implement periodic garbage collection',
                    'Profile and fix memory leaks'
                ],
                risk_assessment='Low risk, memory stability improvement'
            ))

        # GPU optimization recommendations
        if any(b.bottleneck_type.startswith('gpu') for b in bottlenecks):
            recommendations.append(OptimizationRecommendation(
                optimization_type='gpu_optimization',
                target_component='gpu_operations',
                description='Optimize GPU usage with mixed precision and memory management',
                expected_improvement='50-70% improvement in GPU efficiency',
                implementation_complexity='medium',
                priority='high',
                code_changes=[
                    'Enable mixed precision training (FP16)',
                    'Implement gradient accumulation',
                    'Use GPU memory pooling',
                    'Optimize data transfer between CPU/GPU'
                ],
                risk_assessment='Medium risk, requires model validation'
            ))

        # Architecture optimization recommendations
        if metrics_trends.get('degradation_trend', 0) > 0.1:  # 10% degradation
            recommendations.append(OptimizationRecommendation(
                optimization_type='architecture_optimization',
                target_component='system_architecture',
                description='Refactor system architecture for better performance scaling',
                expected_improvement='25-40% overall performance improvement',
                implementation_complexity='high',
                priority='medium',
                code_changes=[
                    'Implement microservices architecture',
                    'Add caching layers for frequent operations',
                    'Optimize data structures and algorithms',
                    'Implement load balancing for distributed processing'
                ],
                risk_assessment='High risk, architectural changes required'
            ))

        # Predictive optimizations based on trends
        if metrics_trends.get('memory_leak_detected', False):
            recommendations.append(OptimizationRecommendation(
                optimization_type='memory_leak_fix',
                target_component='memory_management',
                description='Fix detected memory leaks and implement monitoring',
                expected_improvement='Stable memory usage over time',
                implementation_complexity='low',
                priority='critical',
                code_changes=[
                    'Add memory profiling to identify leaks',
                    'Implement proper object cleanup',
                    'Add memory usage monitoring',
                    'Fix circular references in data structures'
                ],
                risk_assessment='Low risk, stability improvement'
            ))

        self.optimization_recommendations.extend(recommendations)
        return recommendations

    def predict_resource_usage(self, time_horizon: int = 60) -> Dict[str, Any]:
        """Predict future resource usage based on current trends"""
        if len(self.metrics_history) < 10:
            return {'error': 'Insufficient data for prediction'}

        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements

        # Simple linear regression for prediction
        timestamps = np.arange(len(recent_metrics))

        predictions = {}

        # CPU prediction
        cpu_values = [m.cpu_percent for m in recent_metrics]
        cpu_slope, cpu_intercept = np.polyfit(timestamps, cpu_values, 1)
        cpu_prediction = cpu_intercept + cpu_slope * time_horizon
        predictions['cpu_percent'] = {
            'current': cpu_values[-1],
            'predicted': max(0, min(100, cpu_prediction)),
            'trend': 'increasing' if cpu_slope > 0.1 else 'decreasing' if cpu_slope < -0.1 else 'stable',
            'confidence': 0.8
        }

        # Memory prediction
        memory_values = [m.memory_percent for m in recent_metrics]
        memory_slope, memory_intercept = np.polyfit(timestamps, memory_values, 1)
        memory_prediction = memory_intercept + memory_slope * time_horizon
        predictions['memory_percent'] = {
            'current': memory_values[-1],
            'predicted': max(0, min(100, memory_prediction)),
            'trend': 'increasing' if memory_slope > 0.1 else 'decreasing' if memory_slope < -0.1 else 'stable',
            'confidence': 0.8
        }

        # Resource pre-allocation recommendations
        pre_allocation = self._generate_pre_allocation_recommendations(predictions)

        return {
            'predictions': predictions,
            'time_horizon_minutes': time_horizon,
            'pre_allocation_recommendations': pre_allocation,
            'alerts': self._generate_prediction_alerts(predictions)
        }

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)

            # Check for critical thresholds
            self._check_critical_thresholds(metrics)

            time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics"""
        process = psutil.Process()

        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used // (1024 * 1024)  # MB

        # Thread and file information
        active_threads = threading.active_count()
        try:
            open_files = len(process.open_files())
        except:
            open_files = 0

        try:
            network_connections = len(process.connections())
        except:
            network_connections = 0

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read = disk_io.read_bytes if disk_io else 0
        disk_io_write = disk_io.write_bytes if disk_io else 0

        # System-wide context switches and page faults
        try:
            context_switches = psutil.cpu_stats().ctx_switches
            page_faults = process.memory_info().num_page_faults
        except:
            context_switches = 0
            page_faults = 0

        # GPU metrics (if available)
        gpu_memory_used = None
        gpu_memory_total = None

        try:
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() // (1024 * 1024)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except:
            pass

        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            active_threads=active_threads,
            open_files=open_files,
            network_connections=network_connections,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            context_switches=context_switches,
            page_faults=page_faults
        )

    def _record_function_profile(self, func: Callable, execution_time: float, memory_delta: int):
        """Record function execution profile"""
        func_name = f"{func.__module__}.{func.__name__}"
        module_name = func.__module__

        if func_name not in self.function_profiles:
            self.function_profiles[func_name] = FunctionProfile(
                function_name=func.__name__,
                module_name=module_name
            )

        profile = self.function_profiles[func_name]
        profile.call_count += 1
        profile.total_time += execution_time
        profile.avg_time = profile.total_time / profile.call_count
        profile.max_time = max(profile.max_time, execution_time)
        profile.min_time = min(profile.min_time, execution_time)
        profile.memory_delta += memory_delta
        profile.last_called = datetime.now()

        # Check for bottlenecks
        if execution_time > self.thresholds['response_time']['critical']:
            profile.bottlenecks.append(f"Critical execution time: {execution_time:.2f}s")
        elif execution_time > self.thresholds['response_time']['warning']:
            profile.bottlenecks.append(f"High execution time: {execution_time:.2f}s")

        if abs(memory_delta) > 100:  # 100MB delta
            profile.bottlenecks.append(f"High memory delta: {memory_delta:+d}MB")

    def _analyze_profiling_results(self, profile_output: str) -> Dict[str, Any]:
        """Analyze profiling output for insights"""
        analysis = {
            'top_functions': [],
            'total_calls': 0,
            'total_time': 0.0,
            'bottlenecks': []
        }

        lines = profile_output.split('\n')
        in_stats = False

        for line in lines:
            if line.startswith('   ncalls'):
                in_stats = True
                continue
            elif in_stats and line.strip():
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        ncalls = int(parts[0].split('/')[0])  # Handle recursive calls
                        tottime = float(parts[1])
                        percall = float(parts[2])
                        cumtime = float(parts[3])
                        percall_cum = float(parts[4])
                        filename_lineno_func = ' '.join(parts[5:])

                        analysis['top_functions'].append({
                            'function': filename_lineno_func,
                            'calls': ncalls,
                            'total_time': tottime,
                            'cumulative_time': cumtime,
                            'avg_time': percall
                        })

                        analysis['total_calls'] += ncalls
                        analysis['total_time'] += tottime

                except (ValueError, IndexError):
                    continue

        # Identify bottlenecks from profiling
        if analysis['top_functions']:
            # Sort by cumulative time
            analysis['top_functions'].sort(key=lambda x: x['cumulative_time'], reverse=True)

            # Find functions taking >10% of total time
            threshold = analysis['total_time'] * 0.1
            bottlenecks = [f for f in analysis['top_functions'][:5] if f['cumulative_time'] > threshold]
            analysis['bottlenecks'] = bottlenecks

        return analysis

    def _analyze_function_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze function-level bottlenecks"""
        bottlenecks = []

        for func_name, profile in self.function_profiles.items():
            if profile.avg_time > self.thresholds['response_time']['critical']:
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type='function_performance',
                    severity='critical',
                    description=f"Function {func_name} has critical average execution time: {profile.avg_time:.2f}s",
                    impact_score=0.8,
                    recommendations=[
                        f"Optimize {func_name} implementation",
                        "Consider caching results if function is called frequently",
                        "Profile internal function calls for hotspots"
                    ],
                    estimated_fix_time='2-6 hours',
                    confidence=0.9
                ))
            elif profile.avg_time > self.thresholds['response_time']['warning']:
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type='function_performance',
                    severity='high',
                    description=f"Function {func_name} has high average execution time: {profile.avg_time:.2f}s",
                    impact_score=0.6,
                    recommendations=[
                        f"Review {func_name} algorithm complexity",
                        "Consider memoization for repeated calls",
                        "Profile function for optimization opportunities"
                    ],
                    estimated_fix_time='1-3 hours',
                    confidence=0.8
                ))

        return bottlenecks

    def _analyze_metrics_trends(self) -> Dict[str, Any]:
        """Analyze trends in performance metrics"""
        if len(self.metrics_history) < 5:
            return {}

        recent_metrics = list(self.metrics_history)[-20:]

        # Calculate trends
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]

        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]

        # Memory leak detection
        memory_leak_rate = memory_trend * 60  # Per minute
        memory_leak_detected = memory_leak_rate > self.thresholds['memory_leak_rate']['warning']

        # Overall degradation trend
        degradation_score = (cpu_trend + memory_trend) / 200  # Normalized

        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'memory_leak_rate': memory_leak_rate,
            'memory_leak_detected': memory_leak_detected,
            'degradation_trend': degradation_score,
            'trend_period_minutes': len(recent_metrics)
        }

    def _check_critical_thresholds(self, metrics: PerformanceMetrics):
        """Check for critical performance thresholds and alert"""
        alerts = []

        if metrics.cpu_percent > self.thresholds['cpu_percent']['critical']:
            alerts.append(f"CRITICAL: CPU usage at {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.thresholds['cpu_percent']['warning']:
            alerts.append(f"WARNING: High CPU usage at {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.thresholds['memory_percent']['critical']:
            alerts.append(f"CRITICAL: Memory usage at {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.thresholds['memory_percent']['warning']:
            alerts.append(f"WARNING: High memory usage at {metrics.memory_percent:.1f}%")

        if alerts:
            print(f"🚨 Performance Alert: {'; '.join(alerts)}")

    def _average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average metrics from a list"""
        if not metrics_list:
            return PerformanceMetrics(timestamp=datetime.now(), cpu_percent=0, memory_percent=0, memory_used=0)

        avg_cpu = np.mean([m.cpu_percent for m in metrics_list])
        avg_memory_percent = np.mean([m.memory_percent for m in metrics_list])
        avg_memory_used = int(np.mean([m.memory_used for m in metrics_list]))
        avg_threads = int(np.mean([m.active_threads for m in metrics_list]))

        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=avg_cpu,
            memory_percent=avg_memory_percent,
            memory_used=avg_memory_used,
            active_threads=avg_threads
        )

    def _generate_pre_allocation_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate resource pre-allocation recommendations"""
        recommendations = []

        cpu_pred = predictions.get('cpu_percent', {})
        memory_pred = predictions.get('memory_percent', {})

        if cpu_pred.get('predicted', 0) > self.thresholds['cpu_percent']['warning']:
            recommendations.append("Pre-allocate CPU resources for predicted high usage")
            recommendations.append("Consider scaling up CPU capacity in advance")

        if memory_pred.get('predicted', 0) > self.thresholds['memory_percent']['warning']:
            recommendations.append("Pre-allocate memory buffers for predicted high usage")
            recommendations.append("Consider memory optimization before peak usage")

        if cpu_pred.get('trend') == 'increasing' or memory_pred.get('trend') == 'increasing':
            recommendations.append("Monitor resource trends closely - usage is increasing")

        return recommendations

    def _generate_prediction_alerts(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate alerts based on predictions"""
        alerts = []

        for metric_name, pred in predictions.items():
            predicted_value = pred.get('predicted', 0)
            threshold = self.thresholds.get(metric_name.replace('_percent', ''), {})

            if predicted_value > threshold.get('critical', 100):
                alerts.append(f"CRITICAL: {metric_name} predicted to reach {predicted_value:.1f}%")
            elif predicted_value > threshold.get('warning', 100):
                alerts.append(f"WARNING: {metric_name} predicted to reach {predicted_value:.1f}%")

        return alerts

    def record_event(self, event_name: str, timestamp: float):
        """Record a performance-related event or bottleneck for analysis"""
        logger.info(f"Performance event recorded: {event_name} at {timestamp}")
        # Add to function profiles as a dummy if needed, or just log
        if "events" not in self.function_profiles:
            self.function_profiles["events"] = FunctionProfile("events", "performance")
        self.function_profiles["events"].bottlenecks.append(f"{event_name} at {timestamp}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {'error': 'No performance data available'}

        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements

        report = {
            'monitoring_status': 'active' if self.is_monitoring else 'inactive',
            'data_points': len(self.metrics_history),
            'time_range': {
                'start': self.metrics_history[0].timestamp.isoformat(),
                'end': self.metrics_history[-1].timestamp.isoformat()
            },
            'current_metrics': self._collect_metrics().__dict__,
            'averages': {
                'cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
                'memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
                'active_threads': np.mean([m.active_threads for m in recent_metrics])
            },
            'peaks': {
                'cpu_percent': max(m.cpu_percent for m in recent_metrics),
                'memory_percent': max(m.memory_percent for m in recent_metrics),
                'active_threads': max(m.active_threads for m in recent_metrics)
            },
            'bottlenecks': [b.__dict__ for b in self.analyze_bottlenecks()],
            'optimization_recommendations': [r.__dict__ for r in self.generate_optimization_recommendations()],
            'resource_predictions': self.predict_resource_usage(),
            'function_profiles': {name: profile.__dict__ for name, profile in self.function_profiles.items()},
            'system_health_score': self._calculate_health_score()
        }

        return report

    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        if not self.metrics_history:
            return 50.0

        recent_metrics = list(self.metrics_history)[-20:]
        score = 100.0

        # CPU health
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        if avg_cpu > 90:
            score -= 30
        elif avg_cpu > 70:
            score -= 15

        # Memory health
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        if avg_memory > 95:
            score -= 30
        elif avg_memory > 80:
            score -= 15

        # Thread health (too many threads can indicate problems)
        avg_threads = np.mean([m.active_threads for m in recent_metrics])
        if avg_threads > 100:
            score -= 20
        elif avg_threads > 50:
            score -= 10

        return max(0.0, min(100.0, score))


# Global profiler instance
_global_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def profile_function(func: Callable) -> Callable:
    """Decorator to profile function performance"""
    profiler = get_profiler()
    return profiler.profile_function(func)

def start_performance_monitoring():
    """Start global performance monitoring"""
    profiler = get_profiler()
    profiler.start_monitoring()

def stop_performance_monitoring():
    """Stop global performance monitoring"""
    profiler = get_profiler()
    profiler.stop_monitoring()

def get_performance_report():
    """Get comprehensive performance report"""
    profiler = get_profiler()
    return profiler.get_performance_report()


