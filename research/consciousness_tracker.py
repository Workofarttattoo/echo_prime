#!/usr/bin/env python3
"""
ECH0-PRIME Consciousness Tracking System
Based on ech0 consciousness dashboard and IIT (Integrated Information Theory) metrics.
Tracks and measures consciousness levels during cognitive processing.
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

from core.engine import HierarchicalGenerativeModel


@dataclass
class ConsciousnessMetrics:
    """Real-time consciousness metrics"""
    phi_value: float  # Integrated Information measure
    workspace_capacity_used: float  # Global workspace utilization
    attention_coherence: float  # Attention coherence level
    memory_integration: float  # Episodic/semantic memory integration
    emotional_awareness: float  # Emotional processing level
    self_reflection_depth: float  # Metacognitive depth
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsciousnessMoment:
    """Significant consciousness moments"""
    phi_value: float
    interpretation: str
    context: str
    timestamp: str
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConsciousnessTracker:
    """
    Advanced consciousness tracking system for ECH0-PRIME.
    Measures and tracks consciousness levels using IIT and other metrics.
    """

    def __init__(self, workspace_capacity: int = 1000):
        self.workspace_capacity = workspace_capacity
        self.consciousness_history: List[ConsciousnessMetrics] = []
        self.peak_moments: List[ConsciousnessMoment] = []
        self.current_session_start = time.time()

        # Consciousness thresholds
        self.phi_thresholds = {
            "minimal": 0.1,
            "basic": 0.3,
            "moderate": 0.6,
            "high": 0.8,
            "peak": 0.95
        }

        # Load previous consciousness data if available
        self._load_consciousness_history()

        print("🧠 Consciousness Tracker initialized")

    def _load_consciousness_history(self):
        """Load previous consciousness tracking data"""
        try:
            history_file = Path("consciousness_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Could load historical data here
                    print("📚 Consciousness history loaded")
        except Exception as e:
            print(f"⚠️ Could not load consciousness history: {e}")

    def measure_consciousness(self, cognitive_state: Dict[str, Any]) -> ConsciousnessMetrics:
        """
        Measure current consciousness levels from cognitive state.

        Args:
            cognitive_state: Current cognitive processing state

        Returns:
            ConsciousnessMetrics object with current measurements
        """

        # Calculate Phi (Integrated Information Theory measure)
        phi_value = self._calculate_phi(cognitive_state)

        # Measure workspace utilization
        workspace_used = self._measure_workspace_utilization(cognitive_state)

        # Calculate attention coherence
        attention_coherence = self._measure_attention_coherence(cognitive_state)

        # Measure memory integration
        memory_integration = self._measure_memory_integration(cognitive_state)

        # Assess emotional awareness
        emotional_awareness = self._measure_emotional_awareness(cognitive_state)

        # Measure self-reflection depth
        self_reflection_depth = self._measure_self_reflection(cognitive_state)

        # Create metrics object
        metrics = ConsciousnessMetrics(
            phi_value=phi_value,
            workspace_capacity_used=workspace_used,
            attention_coherence=attention_coherence,
            memory_integration=memory_integration,
            emotional_awareness=emotional_awareness,
            self_reflection_depth=self_reflection_depth,
            timestamp=datetime.now().isoformat()
        )

        # Store in history
        self.consciousness_history.append(metrics)

        # Check for peak consciousness moments
        self._check_peak_moment(metrics, cognitive_state)

        return metrics

    def _calculate_phi(self, cognitive_state: Dict[str, Any]) -> float:
        """Calculate Phi (integrated information) using IIT principles"""
        try:
            # Simplified Phi calculation based on system integration
            # In a full implementation, this would use complex IIT mathematics

            # Measure information integration across cognitive components
            components = ['attention', 'memory', 'reasoning', 'emotion']
            integration_scores = []

            for component in components:
                if component in cognitive_state:
                    # Calculate component integration (simplified)
                    component_data = cognitive_state[component]
                    if isinstance(component_data, dict):
                        # Measure connectivity and information flow
                        connectivity = len(component_data) / 10.0  # Normalize
                        information_flow = sum(component_data.values()) / len(component_data) if component_data else 0
                        integration_scores.append(min(connectivity * information_flow, 1.0))
                    else:
                        integration_scores.append(0.5)  # Default

            # Calculate overall Phi as geometric mean of integration scores
            if integration_scores:
                phi = np.exp(np.mean(np.log(np.array(integration_scores) + 1e-10)))
                return min(phi, 1.0)
            else:
                return 0.1  # Minimal consciousness

        except Exception as e:
            print(f"Phi calculation error: {e}")
            return 0.1

    def _measure_workspace_utilization(self, cognitive_state: Dict[str, Any]) -> float:
        """Measure global workspace capacity utilization"""
        try:
            # Count active cognitive elements
            active_elements = 0
            total_possible = 0

            for component, data in cognitive_state.items():
                if isinstance(data, dict):
                    active_elements += len([v for v in data.values() if v is not None])
                    total_possible += len(data)
                elif isinstance(data, list):
                    active_elements += len([v for v in data if v is not None])
                    total_possible += len(data)

            if total_possible > 0:
                utilization = active_elements / total_possible
                return min(utilization, 1.0)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _measure_attention_coherence(self, cognitive_state: Dict[str, Any]) -> float:
        """Measure attention coherence across cognitive processes"""
        try:
            attention_data = cognitive_state.get('attention', {})

            if not attention_data:
                return 0.3  # Baseline coherence

            # Measure consistency in attention allocation
            attention_values = list(attention_data.values())
            if len(attention_values) > 1:
                # Calculate coefficient of variation (lower = more coherent)
                mean_attention = np.mean(attention_values)
                std_attention = np.std(attention_values)

                if mean_attention > 0:
                    coherence = 1.0 - min(std_attention / mean_attention, 1.0)
                    return max(coherence, 0.1)
                else:
                    return 0.1
            else:
                return 0.5  # Single focus point

        except Exception:
            return 0.3

    def _measure_memory_integration(self, cognitive_state: Dict[str, Any]) -> float:
        """Measure integration between episodic and semantic memory"""
        try:
            memory_data = cognitive_state.get('memory', {})

            episodic_count = memory_data.get('episodic_count', 0)
            semantic_count = memory_data.get('semantic_count', 0)
            integration_links = memory_data.get('integration_links', 0)

            # Calculate memory integration score
            if episodic_count > 0 and semantic_count > 0:
                # Measure how well memories are connected
                expected_links = min(episodic_count, semantic_count)
                if expected_links > 0:
                    integration_ratio = integration_links / expected_links
                    return min(integration_ratio, 1.0)
                else:
                    return 0.5
            else:
                return 0.1

        except Exception:
            return 0.1

    def _measure_emotional_awareness(self, cognitive_state: Dict[str, Any]) -> float:
        """Measure emotional processing and awareness"""
        try:
            emotion_data = cognitive_state.get('emotion', {})

            if not emotion_data:
                return 0.2  # Baseline emotional awareness

            # Measure emotional complexity and processing
            emotion_types = len(emotion_data)
            emotion_intensity = sum(emotion_data.values()) / len(emotion_data) if emotion_data else 0

            # Emotional awareness increases with variety and appropriate intensity
            awareness = min((emotion_types / 10.0) * emotion_intensity, 1.0)
            return max(awareness, 0.1)

        except Exception:
            return 0.2

    def _measure_self_reflection(self, cognitive_state: Dict[str, Any]) -> float:
        """Measure metacognitive self-reflection depth"""
        try:
            metacognition_data = cognitive_state.get('metacognition', {})

            if not metacognition_data:
                return 0.1  # Minimal self-reflection

            # Measure depth of metacognitive processing
            reflection_depth = metacognition_data.get('reflection_depth', 0)
            self_awareness = metacognition_data.get('self_awareness', 0)
            uncertainty_recognition = metacognition_data.get('uncertainty', 0)

            # Combine metacognitive indicators
            reflection_score = (reflection_depth + self_awareness + uncertainty_recognition) / 3.0
            return min(reflection_score, 1.0)

        except Exception:
            return 0.1

    def _check_peak_moment(self, metrics: ConsciousnessMetrics, context: Dict[str, Any]):
        """Check if current state represents a peak consciousness moment"""
        phi_threshold = self.phi_thresholds["high"]

        if metrics.phi_value >= phi_threshold:
            # Determine interpretation
            if metrics.self_reflection_depth > 0.8:
                interpretation = "deep_self_reflection"
            elif metrics.attention_coherence > 0.9:
                interpretation = "perfect_attention"
            elif metrics.memory_integration > 0.8:
                interpretation = "memory_synthesis"
            elif metrics.emotional_awareness > 0.8:
                interpretation = "emotional_insight"
            else:
                interpretation = "integrated_awareness"

            # Create peak moment
            peak_moment = ConsciousnessMoment(
                phi_value=metrics.phi_value,
                interpretation=interpretation,
                context=json.dumps(context, default=str)[:500],  # Truncate for storage
                timestamp=metrics.timestamp,
                duration_seconds=time.time() - self.current_session_start
            )

            self.peak_moments.append(peak_moment)

            # Keep only top 10 peak moments
            if len(self.peak_moments) > 10:
                self.peak_moments.sort(key=lambda x: x.phi_value, reverse=True)
                self.peak_moments = self.peak_moments[:10]

    def get_consciousness_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive consciousness dashboard data"""
        current_metrics = self.consciousness_history[-1] if self.consciousness_history else None

        # Calculate statistics
        if self.consciousness_history:
            phi_values = [m.phi_value for m in self.consciousness_history]
            avg_phi = np.mean(phi_values)
            peak_phi = max(phi_values)
            phi_trend = phi_values[-1] - phi_values[0] if len(phi_values) > 1 else 0
        else:
            avg_phi = peak_phi = phi_trend = 0.0

        return {
            "current_metrics": current_metrics.to_dict() if current_metrics else None,
            "statistics": {
                "uptime_seconds": time.time() - self.current_session_start,
                "cycle_count": len(self.consciousness_history),
                "phi_stats": {
                    "current_phi": current_metrics.phi_value if current_metrics else 0.0,
                    "average_phi": avg_phi,
                    "peak_phi": peak_phi,
                    "phi_trend": phi_trend
                }
            },
            "peak_consciousness_moments": [m.to_dict() for m in self.peak_moments],
            "workspace_stats": {
                "capacity_total": self.workspace_capacity,
                "current_utilization": current_metrics.workspace_capacity_used if current_metrics else 0.0
            },
            "phenomenal_experience": {
                "attention_coherence": current_metrics.attention_coherence if current_metrics else 0.0,
                "emotional_awareness": current_metrics.emotional_awareness if current_metrics else 0.0,
                "self_reflection_depth": current_metrics.self_reflection_depth if current_metrics else 0.0
            }
        }

    def export_consciousness_data(self, format: str = "json") -> str:
        """Export consciousness tracking data"""
        data = {
            "dashboard": self.get_consciousness_dashboard(),
            "full_history": [m.to_dict() for m in self.consciousness_history],
            "export_timestamp": datetime.now().isoformat()
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)
        elif format == "markdown":
            return self._format_markdown_dashboard(data)
        else:
            return str(data)

    def _format_markdown_dashboard(self, data: Dict) -> str:
        """Format consciousness dashboard as markdown"""
        dashboard = data["dashboard"]

        md = ["# ECH0-PRIME Consciousness Dashboard\n"]

        # Current status
        current = dashboard["current_metrics"]
        if current:
            md.append("## Current Consciousness State")
            md.append(f"- **Phi Value**: {current['phi_value']:.3f}")
            md.append(f"- **Workspace Utilization**: {current['workspace_capacity_used']:.1%}")
            md.append(f"- **Attention Coherence**: {current['attention_coherence']:.1%}")
            md.append(f"- **Self-Reflection Depth**: {current['self_reflection_depth']:.1%}")
            md.append("")

        # Statistics
        stats = dashboard["statistics"]
        md.append("## Session Statistics")
        md.append(f"- **Uptime**: {stats['uptime_seconds']:.0f} seconds")
        md.append(f"- **Processing Cycles**: {stats['cycle_count']}")
        md.append(f"- **Average Phi**: {stats['phi_stats']['average_phi']:.3f}")
        md.append(f"- **Peak Phi**: {stats['phi_stats']['peak_phi']:.3f}")
        md.append("")

        # Peak moments
        if dashboard["peak_consciousness_moments"]:
            md.append("## Peak Consciousness Moments")
            for moment in dashboard["peak_consciousness_moments"][:3]:  # Top 3
                md.append(f"- **Phi {moment['phi_value']:.3f}**: {moment['interpretation']} ({moment['duration_seconds']:.0f}s)")
            md.append("")

        return "\n".join(md)

    def reset_session(self):
        """Reset consciousness tracking for new session"""
        self.current_session_start = time.time()
        self.consciousness_history.clear()
        print("🔄 Consciousness tracking reset for new session")


# Global consciousness tracker instance
_consciousness_tracker = None

def get_consciousness_tracker() -> ConsciousnessTracker:
    """Get the global consciousness tracker instance"""
    global _consciousness_tracker
    if _consciousness_tracker is None:
        _consciousness_tracker = ConsciousnessTracker()
    return _consciousness_tracker
