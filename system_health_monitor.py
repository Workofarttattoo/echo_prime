#!/usr/bin/env python3
"""
ECH0-PRIME System Health Monitoring
Real-time health monitoring with auto-healing and predictive maintenance
"""

import time
import psutil
import threading
import subprocess
import signal
import os
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import gc
import sys
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import socket


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"
    MAINTENANCE = "maintenance"


class ComponentStatus(Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_name: str
    status: ComponentStatus
    health_score: float  # 0-100
    last_check: datetime
    error_count: int = 0
    warning_count: int = 0
    recovery_attempts: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class HealthAlert:
    """System health alert"""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    auto_healing_attempted: bool = False
    healing_successful: bool = False


@dataclass
class AutoHealingAction:
    """Automated healing action"""
    action_id: str
    component: str
    action_type: str
    description: str
    executed_at: datetime
    success: bool
    duration: float
    impact_assessment: str
    rollback_available: bool


@dataclass
class PredictiveMaintenance:
    """Predictive maintenance recommendation"""
    component: str
    failure_probability: float
    time_to_failure: timedelta
    recommended_action: str
    urgency: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    cost_benefit_ratio: float


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring with auto-healing capabilities
    """

    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.healing_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="healing")

        # Component health tracking
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_history = defaultdict(lambda: deque(maxlen=100))

        # Alert system
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        self.alert_handlers: Dict[str, Callable] = {}

        # Auto-healing system
        self.healing_actions: List[AutoHealingAction] = []
        self.healing_strategies: Dict[str, List[Dict[str, Any]]] = self._initialize_healing_strategies()

        # Predictive maintenance
        self.maintenance_predictions: List[PredictiveMaintenance] = []
        self.failure_patterns = self._load_failure_patterns()

        # Graceful degradation
        self.degradation_strategies: Dict[str, List[Dict[str, Any]]] = self._initialize_degradation_strategies()

        # Health thresholds
        self.health_thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90, 'max': 100},
            'memory_usage': {'warning': 80, 'critical': 95, 'max': 100},
            'disk_usage': {'warning': 85, 'critical': 95, 'max': 100},
            'response_time': {'warning': 2.0, 'critical': 5.0, 'max': 10.0},
            'error_rate': {'warning': 0.05, 'critical': 0.15, 'max': 1.0},
            'uptime': {'warning': 0.95, 'critical': 0.85, 'max': 1.0}
        }

        # System components to monitor
        self.system_components = [
            'cpu', 'memory', 'disk', 'network', 'gpu', 'processes',
            'mathematical_verifier', 'pattern_recognizer', 'reasoning_engine',
            'memory_manager', 'voice_synthesis', 'vision_processing'
        ]

        # Initialize component health
        for component in self.system_components:
            self.component_health[component] = ComponentHealth(
                component_name=component,
                status=ComponentStatus.OPERATIONAL,
                health_score=100.0,
                last_check=datetime.now()
            )

    def start_monitoring(self):
        """Start comprehensive health monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # Start predictive maintenance monitoring
        self._start_predictive_monitoring()

        print("🏥 System Health Monitoring Started")

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.healing_executor.shutdown(wait=True)
        print("🛑 System Health Monitoring Stopped")

    def register_component(self, component_name: str, health_check_func: Callable,
                          dependencies: List[str] = None) -> bool:
        """Register a component for health monitoring"""
        if component_name in self.component_health:
            return False

        self.component_health[component_name] = ComponentHealth(
            component_name=component_name,
            status=ComponentStatus.OPERATIONAL,
            health_score=100.0,
            last_check=datetime.now(),
            dependencies=dependencies or []
        )

        # Store the health check function
        setattr(self, f'_check_{component_name}', health_check_func)

        return True

    def register_alert_handler(self, alert_type: str, handler_func: Callable):
        """Register a handler for specific alert types"""
        self.alert_handlers[alert_type] = handler_func

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        overall_health = self._calculate_overall_health()

        return {
            'overall_health': overall_health,
            'overall_status': self._health_score_to_status(overall_health['score']),
            'timestamp': datetime.now().isoformat(),
            'components': {name: self._component_to_dict(comp) for name, comp in self.component_health.items()},
            'active_alerts': [alert.__dict__ for alert in self.active_alerts.values()],
            'recent_healing_actions': [action.__dict__ for action in self.healing_actions[-5:]],
            'predictive_maintenance': [pred.__dict__ for pred in self.maintenance_predictions[:3]],
            'system_metrics': self._get_system_metrics(),
            'health_trends': self._analyze_health_trends()
        }

    def trigger_manual_healing(self, component: str, issue_type: str) -> Dict[str, Any]:
        """Manually trigger healing for a specific component and issue"""
        if component not in self.component_health:
            return {'error': f'Component {component} not registered'}

        healing_strategies = self.healing_strategies.get(issue_type, [])
        if not healing_strategies:
            return {'error': f'No healing strategies available for {issue_type}'}

        # Execute healing strategies
        results = []
        for strategy in healing_strategies:
            result = self._execute_healing_strategy(component, strategy)
            results.append(result)

        return {
            'component': component,
            'issue_type': issue_type,
            'healing_attempts': len(results),
            'successful_healing': any(r['success'] for r in results),
            'results': results
        }

    def graceful_degradation(self, trigger_reason: str) -> Dict[str, Any]:
        """Initiate graceful degradation due to resource constraints"""
        print(f"🔄 Initiating graceful degradation: {trigger_reason}")

        degradation_plan = []
        current_resources = self._assess_resource_availability()

        for strategy_name, strategies in self.degradation_strategies.items():
            for strategy in strategies:
                if self._should_apply_degradation(strategy, current_resources):
                    result = self._apply_degradation_strategy(strategy)
                    degradation_plan.append({
                        'strategy': strategy_name,
                        'applied': result['success'],
                        'impact': result.get('impact', 'unknown'),
                        'recovery_time': result.get('recovery_time', 'unknown')
                    })

        return {
            'trigger_reason': trigger_reason,
            'degradation_applied': len([d for d in degradation_plan if d['applied']]),
            'total_strategies': len(degradation_plan),
            'degradation_plan': degradation_plan,
            'estimated_recovery_time': self._estimate_recovery_time(degradation_plan)
        }

    def _analyze_system_health(self):
        """Analyze system-wide health and detect patterns"""
        # Calculate overall system score
        scores = [c.health_score for c in self.component_health.values()]
        if scores:
            avg_score = sum(scores) / len(scores)
            
            # Simple health state transition
            if avg_score > 90:
                self.health_status = HealthStatus.HEALTHY
            elif avg_score > 60:
                self.health_status = HealthStatus.WARNING
            else:
                self.health_status = HealthStatus.CRITICAL
                
        # Look for resource bottlenecks
        mem_usage = self.component_health.get('memory', ComponentHealth('memory', ComponentStatus.OPERATIONAL, 100, datetime.now())).metrics.get('usage_percent', 0)
        if mem_usage > 90:
            self._raise_alert('system', 'high', f"Critical memory pressure: {mem_usage:.1f}%")
            
        cpu_usage = self.component_health.get('cpu', ComponentHealth('cpu', ComponentStatus.OPERATIONAL, 100, datetime.now())).metrics.get('usage_percent', 0)
        if cpu_usage > 95:
            self._raise_alert('system', 'high', f"Critical CPU load: {cpu_usage:.1f}%")

    def _monitoring_loop(self):
        """Main health monitoring loop"""
        while self.is_monitoring:
            try:
                # Check all components
                for component_name in list(self.component_health.keys()):
                    self._check_component_health(component_name)

                # Analyze system-wide health
                self._analyze_system_health()

                # Check for predictive maintenance needs
                self._update_predictive_maintenance()

                # Auto-healing for critical issues
                self._perform_auto_healing()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                print(f"Health monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _check_component_health(self, component_name: str):
        """Check health of a specific component"""
        component = self.component_health[component_name]

        try:
            # Get health check function
            check_func = getattr(self, f'_check_{component_name}', None)

            if check_func:
                # Custom component check
                health_result = check_func()
            else:
                # Generic system component check
                health_result = self._check_system_component(component_name)

            # Update component health
            component.health_score = health_result['score']
            component.status = self._score_to_status(health_result['score'])
            component.last_check = datetime.now()
            component.metrics.update(health_result.get('metrics', {}))

            # Handle errors/warnings
            if health_result.get('error'):
                component.error_count += 1
                component.last_error = health_result['error']

                if component.error_count >= 3:
                    self._raise_alert(component_name, 'high', f"Repeated errors in {component_name}: {health_result['error']}")

            elif health_result.get('warning'):
                component.warning_count += 1

                if component.warning_count >= 5:
                    self._raise_alert(component_name, 'medium', f"Persistent warnings in {component_name}: {health_result['warning']}")

            # Store in history
            self.health_history[component_name].append({
                'timestamp': datetime.now(),
                'score': component.health_score,
                'status': component.status.value
            })

        except Exception as e:
            component.status = ComponentStatus.FAILED
            component.last_error = str(e)
            self._raise_alert(component_name, 'critical', f"Health check failed for {component_name}: {e}")

    def _check_system_component(self, component_name: str) -> Dict[str, Any]:
        """Check health of system-level components"""
        try:
            if component_name == 'cpu':
                cpu_percent = psutil.cpu_percent(interval=1)
                return {
                    'score': max(0, 100 - cpu_percent),
                    'metrics': {'usage_percent': cpu_percent}
                }

            elif component_name == 'memory':
                memory = psutil.virtual_memory()
                return {
                    'score': max(0, 100 - memory.percent),
                    'metrics': {'used_percent': memory.percent, 'available_mb': memory.available // (1024*1024)}
                }

            elif component_name == 'disk':
                disk = psutil.disk_usage('/')
                return {
                    'score': max(0, 100 - disk.percent),
                    'metrics': {'used_percent': disk.percent, 'free_gb': disk.free // (1024**3)}
                }

            elif component_name == 'network':
                net_io = psutil.net_io_counters()
                return {
                    'score': 95.0,  # Assume healthy unless specific issues
                    'metrics': {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv,
                        'packets_sent': net_io.packets_sent,
                        'packets_recv': net_io.packets_recv
                    }
                }

            elif component_name == 'gpu':
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated()
                        memory_total = torch.cuda.get_device_properties(0).total_memory
                        usage_percent = (memory_used / memory_total) * 100
                        return {
                            'score': max(0, 100 - usage_percent),
                            'metrics': {'used_percent': usage_percent, 'used_mb': memory_used // (1024*1024)}
                        }
                    else:
                        return {'score': 100.0, 'metrics': {'available': False}}
                except:
                    return {'score': 50.0, 'metrics': {'error': 'GPU check failed'}}

            elif component_name == 'processes':
                process_count = len(psutil.pids())
                zombie_count = len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'zombie'])
                score = max(0, 100 - (zombie_count * 10) - (process_count / 10))
                return {
                    'score': score,
                    'metrics': {'total_processes': process_count, 'zombie_processes': zombie_count}
                }

            else:
                # Generic component check - assume healthy
                return {'score': 90.0, 'metrics': {'generic_check': True}}

        except Exception as e:
            return {
                'score': 20.0,
                'error': str(e),
                'metrics': {'check_failed': True}
            }

    def _raise_alert(self, component: str, severity: str, message: str):
        """Raise a health alert"""
        alert_id = f"alert_{component}_{int(time.time())}"

        alert = HealthAlert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now()
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        print(f"🚨 Health Alert [{severity.upper()}]: {component} - {message}")

        # Trigger auto-healing for critical alerts
        if severity in ['high', 'critical']:
            self.healing_executor.submit(self._perform_auto_healing, component, alert)

    def _perform_auto_healing(self, component: str = None, alert: HealthAlert = None):
        """Perform automated healing actions"""
        if alert and alert.component in self.component_health:
            component = alert.component

        if not component or component not in self.healing_strategies:
            return

        component_obj = self.component_health[component]
        healing_strategies = self.healing_strategies[component]

        for strategy in healing_strategies:
            if self._should_apply_healing(strategy, component_obj):
                result = self._execute_healing_strategy(component, strategy)

                # Record healing action
                healing_action = AutoHealingAction(
                    action_id=f"healing_{component}_{int(time.time())}",
                    component=component,
                    action_type=strategy['type'],
                    description=strategy['description'],
                    executed_at=datetime.now(),
                    success=result['success'],
                    duration=result.get('duration', 0.0),
                    impact_assessment=result.get('impact', 'unknown'),
                    rollback_available=strategy.get('rollback_available', False)
                )

                self.healing_actions.append(healing_action)

                if result['success']:
                    print(f"✅ Auto-healing successful: {component} - {strategy['description']}")
                    if alert:
                        alert.auto_healing_attempted = True
                        alert.healing_successful = True
                        alert.resolved = True
                        alert.resolution_time = datetime.now()
                    break
                else:
                    print(f"❌ Auto-healing failed: {component} - {strategy['description']}")

    def _execute_healing_strategy(self, component: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific healing strategy"""
        start_time = time.time()

        try:
            strategy_type = strategy['type']

            if strategy_type == 'restart_component':
                success = self._restart_component(component)
                impact = "Temporary service interruption"

            elif strategy_type == 'memory_cleanup':
                success = self._perform_memory_cleanup()
                impact = "Temporary performance impact during cleanup"

            elif strategy_type == 'resource_reallocation':
                success = self._reallocate_resources(component)
                impact = "Resource redistribution may affect other components"

            elif strategy_type == 'configuration_reset':
                success = self._reset_component_configuration(component)
                impact = "Component configuration reset to defaults"

            elif strategy_type == 'dependency_restart':
                success = self._restart_component_dependencies(component)
                impact = "Multiple component restart may cause broader impact"

            else:
                success = False
                impact = "Unknown strategy type"

            duration = time.time() - start_time

            return {
                'success': success,
                'duration': duration,
                'impact': impact,
                'strategy_type': strategy_type
            }

        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'impact': 'Unknown - healing failed'
            }

    def _restart_component(self, component: str) -> bool:
        """Restart a component"""
        try:
            # This is a simplified restart - in practice would depend on component type
            if component in ['cpu', 'memory', 'disk']:  # System components can't be restarted
                return False

            # For software components, attempt restart through available interfaces
            # This would need to be customized for each component
            print(f"Attempting to restart component: {component}")
            time.sleep(1)  # Simulate restart time

            # Reset component health
            if component in self.component_health:
                comp = self.component_health[component]
                comp.status = ComponentStatus.RECOVERING
                comp.error_count = 0
                comp.recovery_attempts += 1

            return True

        except Exception as e:
            print(f"Component restart failed: {e}")
            return False

    def _perform_memory_cleanup(self) -> bool:
        """Perform memory cleanup"""
        try:
            # Force garbage collection
            collected = gc.collect()

            # Clear any cached data if available
            # This would need to be integrated with the actual caching systems

            print(f"Memory cleanup completed: {collected} objects collected")
            return True

        except Exception as e:
            print(f"Memory cleanup failed: {e}")
            return False

    def _reallocate_resources(self, component: str) -> bool:
        """Reallocate resources to a component"""
        try:
            # This would interact with resource managers
            print(f"Reallocating resources for: {component}")
            return True
        except Exception as e:
            print(f"Resource reallocation failed: {e}")
            return False

    def _reset_component_configuration(self, component: str) -> bool:
        """Reset component configuration to defaults"""
        try:
            print(f"Resetting configuration for: {component}")
            # This would need to interact with configuration management
            return True
        except Exception as e:
            print(f"Configuration reset failed: {e}")
            return False

    def _restart_component_dependencies(self, component: str) -> bool:
        """Restart component dependencies"""
        try:
            comp = self.component_health.get(component)
            if not comp or not comp.dependencies:
                return False

            print(f"Restarting dependencies for {component}: {comp.dependencies}")
            for dep in comp.dependencies:
                self._restart_component(dep)

            return True
        except Exception as e:
            print(f"Dependency restart failed: {e}")
            return False

    def _should_apply_healing(self, strategy: Dict[str, Any], component: ComponentHealth) -> bool:
        """Determine if a healing strategy should be applied"""
        # Check strategy conditions
        conditions = strategy.get('conditions', {})

        if 'min_health_score' in conditions and component.health_score >= conditions['min_health_score']:
            return False

        if 'max_error_count' in conditions and component.error_count <= conditions['max_error_count']:
            return False

        if 'required_status' in conditions and component.status != ComponentStatus(conditions['required_status']):
            return False

        return True

    def _initialize_healing_strategies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize auto-healing strategies for different components"""
        return {
            'cpu': [
                {
                    'type': 'resource_reallocation',
                    'description': 'Reduce CPU-intensive tasks',
                    'conditions': {'min_health_score': 20},
                    'rollback_available': True
                }
            ],
            'memory': [
                {
                    'type': 'memory_cleanup',
                    'description': 'Force garbage collection and clear caches',
                    'conditions': {'min_health_score': 30},
                    'rollback_available': False
                },
                {
                    'type': 'resource_reallocation',
                    'description': 'Reduce memory allocation',
                    'conditions': {'min_health_score': 10},
                    'rollback_available': True
                }
            ],
            'mathematical_verifier': [
                {
                    'type': 'restart_component',
                    'description': 'Restart mathematical verification service',
                    'conditions': {'max_error_count': 5},
                    'rollback_available': True
                },
                {
                    'type': 'configuration_reset',
                    'description': 'Reset verification parameters to defaults',
                    'conditions': {'min_health_score': 50},
                    'rollback_available': True
                }
            ],
            'reasoning_engine': [
                {
                    'type': 'restart_component',
                    'description': 'Restart reasoning engine',
                    'conditions': {'max_error_count': 3},
                    'rollback_available': True
                },
                {
                    'type': 'memory_cleanup',
                    'description': 'Clear reasoning cache',
                    'conditions': {'min_health_score': 60},
                    'rollback_available': True
                }
            ],
            'vision_processing': [
                {
                    'type': 'restart_component',
                    'description': 'Restart vision processing pipeline',
                    'conditions': {'max_error_count': 3},
                    'rollback_available': True
                },
                {
                    'type': 'resource_reallocation',
                    'description': 'Reallocate GPU resources',
                    'conditions': {'min_health_score': 40},
                    'rollback_available': True
                }
            ]
        }

    def _initialize_degradation_strategies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize graceful degradation strategies"""
        return {
            'reduce_precision': [
                {
                    'description': 'Reduce numerical precision to save memory',
                    'trigger_conditions': {'memory_percent': 85},
                    'impact': 'Slight accuracy reduction',
                    'recovery_time': 'immediate'
                }
            ],
            'disable_features': [
                {
                    'description': 'Disable non-essential features',
                    'trigger_conditions': {'cpu_percent': 80},
                    'impact': 'Reduced functionality',
                    'recovery_time': 'on_demand'
                }
            ],
            'reduce_batch_size': [
                {
                    'description': 'Reduce processing batch sizes',
                    'trigger_conditions': {'memory_percent': 90},
                    'impact': 'Slower processing',
                    'recovery_time': 'immediate'
                }
            ],
            'simplify_models': [
                {
                    'description': 'Use simplified model versions',
                    'trigger_conditions': {'memory_percent': 95},
                    'impact': 'Reduced accuracy',
                    'recovery_time': 'on_restart'
                }
            ]
        }

    def _should_apply_degradation(self, strategy: Dict[str, Any], resources: Dict[str, Any]) -> bool:
        """Determine if a degradation strategy should be applied"""
        conditions = strategy.get('trigger_conditions', {})

        for metric, threshold in conditions.items():
            if metric in resources and resources[metric] > threshold:
                return True

        return False

    def _apply_degradation_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a graceful degradation strategy"""
        try:
            description = strategy['description']

            # This would implement actual degradation logic
            print(f"Applying degradation strategy: {description}")

            # Simulate successful application
            return {
                'success': True,
                'impact': strategy['impact'],
                'recovery_time': strategy['recovery_time']
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'impact': 'unknown'
            }

    def _estimate_recovery_time(self, degradation_plan: List[Dict[str, Any]]) -> str:
        """Estimate time to recover from degradation"""
        recovery_times = []

        for item in degradation_plan:
            recovery_time = item.get('recovery_time', 'unknown')
            if recovery_time != 'unknown':
                recovery_times.append(recovery_time)

        if 'on_restart' in recovery_times:
            return 'on_system_restart'
        elif 'on_demand' in recovery_times:
            return 'on_demand'
        elif recovery_times:
            return 'immediate'
        else:
            return 'unknown'

    def _start_predictive_monitoring(self):
        """Start predictive maintenance monitoring"""
        # This would run in a separate thread to analyze trends
        pass

    def _update_predictive_maintenance(self):
        """Update predictive maintenance predictions"""
        # Analyze component health trends to predict failures
        for component_name, component in self.component_health.items():
            if len(self.health_history[component_name]) >= 10:
                health_scores = [h['score'] for h in self.health_history[component_name]]

                # Simple trend analysis
                if len(health_scores) >= 5:
                    recent_trend = np.polyfit(range(len(health_scores)), health_scores, 1)[0]

                    if recent_trend < -2:  # Health declining
                        failure_prob = min(0.9, abs(recent_trend) / 10)
                        time_to_failure = timedelta(hours=max(1, 100 / abs(recent_trend)))

                        prediction = PredictiveMaintenance(
                            component=component_name,
                            failure_probability=failure_prob,
                            time_to_failure=time_to_failure,
                            recommended_action=f"Monitor {component_name} closely and prepare maintenance",
                            urgency='medium' if failure_prob > 0.5 else 'low',
                            confidence=0.7,
                            cost_benefit_ratio=2.5  # Benefit vs cost of preventive action
                        )

                        self.maintenance_predictions.append(prediction)

                        # Keep only recent predictions
                        if len(self.maintenance_predictions) > 20:
                            self.maintenance_predictions = self.maintenance_predictions[-20:]

    def _load_failure_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load historical failure patterns for predictive analysis"""
        # This would load from a database or configuration file
        return {
            'memory_leak': [
                {'symptoms': ['gradual_memory_increase', 'performance_degradation']},
                {'causes': ['circular_references', 'improper_cleanup']},
                {'solutions': ['implement_weak_references', 'add_cleanup_hooks']}
            ],
            'cpu_overload': [
                {'symptoms': ['high_cpu_usage', 'slow_response_times']},
                {'causes': ['inefficient_algorithms', 'too_many_threads']},
                {'solutions': ['optimize_algorithms', 'reduce_thread_count']}
            ]
        }

    def _calculate_overall_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        if not self.component_health:
            return {'score': 50.0, 'status': HealthStatus.WARNING}

        health_scores = [comp.health_score for comp in self.component_health.values()]
        avg_health = np.mean(health_scores)

        # Weight critical components more heavily
        critical_components = ['cpu', 'memory', 'reasoning_engine']
        critical_scores = [comp.health_score for name, comp in self.component_health.items()
                          if name in critical_components]

        if critical_scores:
            critical_avg = np.mean(critical_scores)
            overall_score = (avg_health * 0.7) + (critical_avg * 0.3)
        else:
            overall_score = avg_health

        status = self._health_score_to_status(overall_score)

        return {
            'score': overall_score,
            'status': status,
            'component_count': len(self.component_health),
            'healthy_components': len([c for c in self.component_health.values() if c.status == ComponentStatus.OPERATIONAL]),
            'degraded_components': len([c for c in self.component_health.values() if c.status == ComponentStatus.DEGRADED]),
            'failed_components': len([c for c in self.component_health.values() if c.status == ComponentStatus.FAILED])
        }

    def _health_score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status"""
        if score >= 80:
            return HealthStatus.HEALTHY
        elif score >= 60:
            return HealthStatus.WARNING
        elif score >= 30:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.DOWN

    def _score_to_status(self, score: float) -> ComponentStatus:
        """Convert health score to component status"""
        if score >= 80:
            return ComponentStatus.OPERATIONAL
        elif score >= 60:
            return ComponentStatus.DEGRADED
        else:
            return ComponentStatus.FAILED

    def _component_to_dict(self, component: ComponentHealth) -> Dict[str, Any]:
        """Convert component health to dictionary"""
        return {
            'name': component.component_name,
            'status': component.status.value,
            'health_score': component.health_score,
            'last_check': component.last_check.isoformat(),
            'error_count': component.error_count,
            'warning_count': component.warning_count,
            'recovery_attempts': component.recovery_attempts,
            'last_error': component.last_error,
            'metrics': component.metrics,
            'dependencies': component.dependencies
        }

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'uptime_seconds': time.time() - psutil.boot_time(),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except:
            return {'error': 'Unable to collect system metrics'}

    def _analyze_health_trends(self) -> Dict[str, Any]:
        """Analyze health trends over time"""
        trends = {}

        for component_name, history in self.health_history.items():
            if len(history) >= 5:
                scores = [h['score'] for h in history]
                trend = np.polyfit(range(len(scores)), scores, 1)[0]

                trends[component_name] = {
                    'current_score': scores[-1],
                    'trend': 'improving' if trend > 0.1 else 'declining' if trend < -0.1 else 'stable',
                    'trend_slope': trend,
                    'data_points': len(scores)
                }

        return trends


# Global health monitor instance
_global_health_monitor = None

def get_health_monitor() -> SystemHealthMonitor:
    """Get the global system health monitor instance"""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = SystemHealthMonitor()
    return _global_health_monitor

def start_health_monitoring():
    """Start global health monitoring"""
    monitor = get_health_monitor()
    monitor.start_monitoring()

def stop_health_monitoring():
    """Stop global health monitoring"""
    monitor = get_health_monitor()
    monitor.stop_monitoring()

def get_system_health():
    """Get comprehensive system health status"""
    monitor = get_health_monitor()
    return monitor.get_system_health()


