"""
Monitoring and observability infrastructure.
"""
import time
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import threading


@dataclass
class Metric:
    """Single metric value"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]


class MetricsCollector:
    """
    Collects metrics from all subsystems.
    """
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def get_metrics(self, name: str, window_seconds: Optional[float] = None) -> List[Metric]:
        """Get metrics for a name, optionally filtered by time window"""
        with self.lock:
            metrics = list(self.metrics[name])
        
        if window_seconds:
            cutoff = time.time() - window_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        return metrics
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        metrics = self.get_metrics(name)
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1]
        }


class DistributedTracing:
    """
    Tracks requests across all components.
    """
    def __init__(self):
        self.traces: Dict[str, List[Dict]] = {}
        self.lock = threading.Lock()
    
    def start_trace(self, trace_id: str, operation: str):
        """Start a trace"""
        with self.lock:
            if trace_id not in self.traces:
                self.traces[trace_id] = []
            
            self.traces[trace_id].append({
                "operation": operation,
                "start_time": time.time(),
                "end_time": None,
                "duration": None
            })
    
    def end_trace(self, trace_id: str):
        """End a trace"""
        with self.lock:
            if trace_id in self.traces and self.traces[trace_id]:
                last_span = self.traces[trace_id][-1]
                last_span["end_time"] = time.time()
                last_span["duration"] = last_span["end_time"] - last_span["start_time"]
    
    def get_trace(self, trace_id: str) -> List[Dict]:
        """Get trace for an ID"""
        with self.lock:
            return self.traces.get(trace_id, [])


class PerformanceProfiler:
    """
    Profiles performance to identify bottlenecks.
    """
    def __init__(self):
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def profile_function(self, func_name: str):
        """Decorator to profile a function"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                with self.lock:
                    self.profiles[func_name].append(duration)
                
                return result
            return wrapper
        return decorator
    
    def get_bottlenecks(self, threshold: float = 1.0) -> List[Dict]:
        """Get functions that take longer than threshold"""
        bottlenecks = []
        
        with self.lock:
            for func_name, durations in self.profiles.items():
                avg_duration = sum(durations) / len(durations) if durations else 0
                if avg_duration > threshold:
                    bottlenecks.append({
                        "function": func_name,
                        "avg_duration": avg_duration,
                        "max_duration": max(durations) if durations else 0,
                        "call_count": len(durations)
                    })
        
        return sorted(bottlenecks, key=lambda x: x["avg_duration"], reverse=True)


class AlertingSystem:
    """
    Notifies on anomalies or failures.
    """
    def __init__(self):
        self.alert_handlers: List[callable] = []
        self.alert_history: deque = deque(maxlen=1000)
    
    def register_handler(self, handler: callable):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
    
    def alert(self, level: str, message: str, context: Optional[Dict] = None):
        """Send an alert"""
        alert = {
            "level": level,
            "message": message,
            "context": context or {},
            "timestamp": time.time()
        }
        
        self.alert_history.append(alert)
        
        # Call all handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")
    
    def check_anomalies(self, metrics_collector: MetricsCollector, thresholds: Dict[str, float]):
        """Check for anomalies in metrics"""
        for metric_name, threshold in thresholds.items():
            summary = metrics_collector.get_summary(metric_name)
            if summary and summary.get("latest", 0) > threshold:
                self.alert(
                    "warning",
                    f"Metric {metric_name} exceeded threshold",
                    {"metric": metric_name, "value": summary["latest"], "threshold": threshold}
                )


class MonitoringSystem:
    """
    Complete monitoring system.
    """
    def __init__(self):
        self.metrics = MetricsCollector()
        self.tracing = DistributedTracing()
        self.profiler = PerformanceProfiler()
        self.alerting = AlertingSystem()
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        all_metrics = {}
        for name in self.metrics.metrics.keys():
            all_metrics[name] = [
                asdict(m) for m in self.metrics.get_metrics(name)
            ]
        
        if format == "json":
            return json.dumps(all_metrics, indent=2)
        else:
            return str(all_metrics)

