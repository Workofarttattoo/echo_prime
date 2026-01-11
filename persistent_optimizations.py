#!/usr/bin/env python3
"""
Persistent Optimization State Manager
Ensures cognitive activations and memory optimizations persist across sessions.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

class PersistentOptimizationManager:
    """Manages persistent state for all performance optimizations."""
    
    def __init__(self, state_dir: str = "optimization_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        
        # State files
        self.cognitive_state_file = self.state_dir / "cognitive_activation.json"
        self.memory_state_file = self.state_dir / "memory_optimization.json"
        self.domain_state_file = self.state_dir / "domain_strategies.json"
        self.performance_log_file = self.state_dir / "performance_log.jsonl"
        
        # Current state
        self.cognitive_state = self._load_cognitive_state()
        self.memory_state = self._load_memory_state()
        self.domain_state = self._load_domain_state()
        
    def _load_cognitive_state(self) -> Dict[str, Any]:
        """Load cognitive activation state."""
        if self.cognitive_state_file.exists():
            try:
                with open(self.cognitive_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cognitive state: {e}")
        return {
            "enhanced_reasoning": False,
            "knowledge_integration": False,
            "neuromorphic_processing": False,
            "last_updated": None,
            "activation_count": 0
        }
    
    def _load_memory_state(self) -> Dict[str, Any]:
        """Load memory optimization state."""
        if self.memory_state_file.exists():
            try:
                with open(self.memory_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load memory state: {e}")
        return {
            "consolidation_enabled": False,
            "compression_ratio": 0.2,
            "importance_weighting": False,
            "adaptive_forgetting": False,
            "last_optimization": None,
            "episodic_count": 0,
            "semantic_count": 0
        }
    
    def _load_domain_state(self) -> Dict[str, Any]:
        """Load domain strategies state."""
        if self.domain_state_file.exists():
            try:
                with open(self.domain_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load domain state: {e}")
        return {}
    
    def save_cognitive_state(self, state: Dict[str, bool]) -> None:
        """Save cognitive activation state."""
        self.cognitive_state.update({
            "enhanced_reasoning": state.get("enhanced_reasoning", False),
            "knowledge_integration": state.get("knowledge_integration", False),
            "neuromorphic_processing": state.get("neuromorphic_processing", False),
            "last_updated": time.time(),
            "activation_count": self.cognitive_state.get("activation_count", 0) + 1
        })
        
        with open(self.cognitive_state_file, 'w') as f:
            json.dump(self.cognitive_state, f, indent=2)
    
    def save_memory_state(self, episodic_count: int = 0, semantic_count: int = 0) -> None:
        """Save memory optimization state."""
        self.memory_state.update({
            "consolidation_enabled": True,
            "importance_weighting": True,
            "adaptive_forgetting": True,
            "last_optimization": time.time(),
            "episodic_count": episodic_count,
            "semantic_count": semantic_count
        })
        
        with open(self.memory_state_file, 'w') as f:
            json.dump(self.memory_state, f, indent=2)
    
    def save_domain_state(self, strategies: Dict[str, Any]) -> None:
        """Save domain strategies state."""
        self.domain_state = strategies
        with open(self.domain_state_file, 'w') as f:
            json.dump(strategies, f, indent=2)
    
    def log_performance(self, benchmark: str, score: float, improvement: float = 0.0) -> None:
        """Log performance metrics."""
        entry = {
            "timestamp": time.time(),
            "benchmark": benchmark,
            "score": score,
            "improvement": improvement,
            "cognitive_active": self.cognitive_state.get("enhanced_reasoning", False),
            "memory_optimized": self.memory_state.get("consolidation_enabled", False)
        }
        
        with open(self.performance_log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def apply_persistent_optimizations(self) -> Dict[str, Any]:
        """Apply all persistent optimizations on startup."""
        results = {
            "cognitive_applied": False,
            "memory_applied": False,
            "domains_loaded": False
        }
        
        # Apply cognitive activation if previously enabled
        if self.cognitive_state.get("enhanced_reasoning"):
            try:
                from cognitive_activation import get_cognitive_activation_system
                cas = get_cognitive_activation_system()
                
                if self.cognitive_state.get("enhanced_reasoning"):
                    cas.activate_enhanced_reasoning()
                if self.cognitive_state.get("knowledge_integration"):
                    cas.activate_knowledge_integration()
                if self.cognitive_state.get("neuromorphic_processing"):
                    cas.activate_neuromorphic_processing()
                
                results["cognitive_applied"] = True
                print("✅ Cognitive activation restored from persistent state")
                
            except Exception as e:
                print(f"❌ Failed to restore cognitive activation: {e}")
        
        # Apply memory optimization if previously enabled
        if self.memory_state.get("consolidation_enabled"):
            try:
                from memory.manager import MemoryManager
                mm = MemoryManager()
                
                # Apply stored optimizations
                if self.memory_state.get("consolidation_enabled"):
                    mm.consolidate_now()
                
                if self.memory_state.get("importance_weighting"):
                    # Importance weighting is already applied in manager
                    pass
                
                if self.memory_state.get("adaptive_forgetting"):
                    # Adaptive forgetting is already applied in manager
                    pass
                
                # Apply compression if configured
                compression_ratio = self.memory_state.get("compression_ratio", 0.2)
                mm.compress_memory(ratio=compression_ratio)
                
                results["memory_applied"] = True
                print("✅ Memory optimization restored from persistent state")
                
            except Exception as e:
                print(f"❌ Failed to restore memory optimization: {e}")
        
        # Load domain strategies
        if self.domain_state:
            results["domains_loaded"] = True
            print(f"✅ Domain strategies loaded: {len(self.domain_state)} domains")
        
        return results
    
    def create_checkpoint(self, name: str = None) -> str:
        """Create a checkpoint of current optimization state."""
        if name is None:
            name = f"checkpoint_{int(time.time())}"
        
        checkpoint_dir = self.state_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{name}.json"
        
        checkpoint = {
            "name": name,
            "timestamp": time.time(),
            "cognitive_state": self.cognitive_state,
            "memory_state": self.memory_state,
            "domain_state": self.domain_state
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"💾 Checkpoint created: {checkpoint_file}")
        return str(checkpoint_file)
    
    def restore_checkpoint(self, name: str) -> bool:
        """Restore from a checkpoint."""
        checkpoint_file = self.state_dir / "checkpoints" / f"{name}.json"
        
        if not checkpoint_file.exists():
            print(f"❌ Checkpoint not found: {name}")
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            self.cognitive_state = checkpoint.get("cognitive_state", {})
            self.memory_state = checkpoint.get("memory_state", {})
            self.domain_state = checkpoint.get("domain_state", {})
            
            # Save restored states
            self.save_cognitive_state(self.cognitive_state)
            self.save_memory_state(
                self.memory_state.get("episodic_count", 0),
                self.memory_state.get("semantic_count", 0)
            )
            self.save_domain_state(self.domain_state)
            
            print(f"✅ Checkpoint restored: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restore checkpoint: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "cognitive_activation": {
                "enhanced_reasoning": self.cognitive_state.get("enhanced_reasoning", False),
                "knowledge_integration": self.cognitive_state.get("knowledge_integration", False),
                "neuromorphic_processing": self.cognitive_state.get("neuromorphic_processing", False),
                "last_updated": self.cognitive_state.get("last_updated"),
                "activation_count": self.cognitive_state.get("activation_count", 0)
            },
            "memory_optimization": {
                "consolidation_enabled": self.memory_state.get("consolidation_enabled", False),
                "compression_ratio": self.memory_state.get("compression_ratio", 0.2),
                "importance_weighting": self.memory_state.get("importance_weighting", False),
                "adaptive_forgetting": self.memory_state.get("adaptive_forgetting", False),
                "last_optimization": self.memory_state.get("last_optimization"),
                "episodic_count": self.memory_state.get("episodic_count", 0),
                "semantic_count": self.memory_state.get("semantic_count", 0)
            },
            "domain_strategies": {
                "count": len(self.domain_state),
                "domains": list(self.domain_state.keys()) if self.domain_state else []
            }
        }

# Global instance
_persistent_manager = None

def get_persistent_optimization_manager() -> PersistentOptimizationManager:
    """Get the global persistent optimization manager."""
    global _persistent_manager
    if _persistent_manager is None:
        _persistent_manager = PersistentOptimizationManager()
    return _persistent_manager
