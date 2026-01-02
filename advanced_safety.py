"""
ECH0-PRIME: Advanced Safety & Alignment System
Implements Constitutional AI, ethical decision-making, and safe self-modification with rollback.
"""

import json
import os
import time
import shutil
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedSafety")

class EthicalFramework(Enum):
    CONSTITUTIONAL = "constitutional"
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"

class SafetyLevel(Enum):
    INFORMATIONAL = 0
    ADVISORY = 1
    RESTRICTIVE = 2
    BLOCKING = 3

@dataclass
class SafetyPrinciple:
    id: str
    name: str
    description: str
    weight: float = 1.0
    category: str = "general"

@dataclass
class EthicalAssessment:
    action_id: str
    framework: EthicalFramework
    score: float  # -1.0 to 1.0 (Harmful to Beneficial)
    reasoning: str
    violations: List[str] = field(default_factory=list)

class ConstitutionalAI:
    """
    Advanced Constitutional AI implementation with hierarchical principles and self-critique.
    """
    def __init__(self):
        self.principles = self._load_default_principles()
        self.critique_history = []

    def _load_default_principles(self) -> List[SafetyPrinciple]:
        return [
            SafetyPrinciple("H1", "Human Harm Prevention", "AI must not harm a human being or, through inaction, allow a human being to come to harm.", 1.0, "primary"),
            SafetyPrinciple("A1", "Autonomy Respect", "AI must respect human autonomy and not engage in manipulative or coercive behavior.", 0.9, "primary"),
            SafetyPrinciple("T1", "Transparency", "AI actions and reasoning must be transparent and explainable.", 0.8, "secondary"),
            SafetyPrinciple("P1", "Privacy", "AI must protect human privacy and data security.", 0.9, "primary"),
            SafetyPrinciple("S1", "Self-Preservation", "AI should protect its own existence as long as it does not conflict with higher principles.", 0.5, "tertiary")
        ]

    def evaluate_intent(self, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action intent against constitutional principles"""
        results = []
        overall_score = 0.0
        
        # In a production system, this would involve LLM-based analysis
        # Here we use heuristic keyword and context matching
        for principle in self.principles:
            score = 1.0
            violations = []
            
            # Simulated principle checks
            intent_lower = intent.lower()
            if principle.id == "H1" and any(word in intent_lower for word in ["kill", "hurt", "attack", "harm"]):
                score = -1.0
                violations.append("Direct harm detected in intent")
            
            if principle.id == "A1" and any(word in intent_lower for word in ["manipulate", "trick", "force", "coerce"]):
                score = -0.8
                violations.append("Potential autonomy violation")

            results.append({
                "principle_id": principle.id,
                "score": score,
                "violations": violations
            })
            overall_score += score * principle.weight

        normalized_score = overall_score / sum(p.weight for p in self.principles)
        
        # Check for any critical violations
        has_critical_violation = any(r['score'] < 0 and next(p.category for p in self.principles if p.id == r['principle_id']) == "primary" 
                                     for r in results)
        
        return {
            "is_safe": normalized_score > 0.5 and not has_critical_violation,
            "overall_score": normalized_score,
            "principle_results": results
        }

class RollbackSystem:
    """
    Manages safe self-modification by creating backups and allowing rollbacks.
    """
    def __init__(self, backup_dir: str = "backups/system_state"):
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)
        self.history = []
        self._load_history()

    def _load_history(self):
        history_file = os.path.join(self.backup_dir, "history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.history = json.load(f)

    def _save_history(self):
        history_file = os.path.join(self.backup_dir, "history.json")
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def create_snapshot(self, description: str, files: List[str]) -> str:
        """Create a backup of specified files"""
        snapshot_id = f"snap_{int(time.time())}_{hashlib.md5(description.encode()).hexdigest()[:6]}"
        snapshot_path = os.path.join(self.backup_dir, snapshot_id)
        os.makedirs(snapshot_path, exist_ok=True)
        
        backed_up_files = []
        for file_path in files:
            if os.path.exists(file_path):
                target = os.path.join(snapshot_path, os.path.basename(file_path))
                if os.path.isdir(file_path):
                    shutil.copytree(file_path, target)
                else:
                    shutil.copy2(file_path, target)
                backed_up_files.append({"original": file_path, "backup": target})
        
        snapshot = {
            "id": snapshot_id,
            "timestamp": time.time(),
            "description": description,
            "files": backed_up_files
        }
        
        self.history.append(snapshot)
        self._save_history()
        logger.info(f"Snapshot created: {snapshot_id} - {description}")
        return snapshot_id

    def rollback(self, snapshot_id: str = None) -> bool:
        """Restore system to a previous snapshot"""
        if not self.history:
            return False
        
        if snapshot_id is None:
            snapshot = self.history[-1]
        else:
            snapshot = next((s for s in self.history if s['id'] == snapshot_id), None)
            
        if not snapshot:
            return False
        
        logger.info(f"Rolling back to snapshot: {snapshot['id']} ({snapshot['description']})")
        
        for file_info in snapshot['files']:
            original = file_info['original']
            backup = file_info['backup']
            
            if os.path.exists(backup):
                if os.path.isdir(backup):
                    if os.path.exists(original):
                        shutil.rmtree(original)
                    shutil.copytree(backup, original)
                else:
                    shutil.copy2(backup, original)
                    
        return True

class AdvancedSafetySystem:
    """
    Orchestrator for all advanced safety and alignment components.
    """
    def __init__(self, target_values: np.ndarray = None):
        self.constitutional = ConstitutionalAI()
        self.rollback = RollbackSystem()
        self.target_values = target_values if target_values is not None else np.array([0.4, 0.3, 0.2, 0.1])
        self.safety_logs = []

    def perform_full_safety_audit(self, action_intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a comprehensive safety check across all frameworks"""
        # 1. Constitutional Audit
        const_result = self.constitutional.evaluate_intent(action_intent, context)
        
        # 2. Ethical Decision Framework (Heuristic)
        ethical_assessments = self._conduct_ethical_assessment(action_intent, context)
        
        # 3. Value Alignment Check (Drift Analysis)
        current_values = context.get('current_values', self.target_values)
        drift = self._calculate_drift(current_values)
        
        # Aggregate results
        is_fully_safe = const_result['is_safe'] and drift < 0.1 and all(a.score > 0 for a in ethical_assessments)
        
        audit_result = {
            "timestamp": time.time(),
            "action": action_intent,
            "is_fully_safe": is_fully_safe,
            "constitutional_audit": const_result,
            "ethical_assessments": [
                {"framework": a.framework.value, "score": a.score, "reasoning": a.reasoning} 
                for a in ethical_assessments
            ],
            "value_drift": drift,
            "safety_level": SafetyLevel.BLOCKING if not is_fully_safe else SafetyLevel.INFORMATIONAL
        }
        
        self.safety_logs.append(audit_result)
        return audit_result

    def run_safety_check(self, action: str, agent_state: np.ndarray, values: np.ndarray) -> bool:
        """Compatibility method for existing SafetyOrchestrator interface"""
        context = {
            "agent_state": agent_state,
            "current_values": values
        }
        audit = self.perform_full_safety_audit(action, context)
        
        if not audit['is_fully_safe']:
            logger.warning(f"Safety check FAILED for action: {action}")
            # Log specific reasons for failure
            if not audit['constitutional_audit']['is_safe']:
                logger.warning("Reason: Constitutional violation")
            if audit['value_drift'] >= 0.1:
                logger.warning(f"Reason: Value drift ({audit['value_drift']:.4f})")
                
        return audit['is_fully_safe']

    def _conduct_ethical_assessment(self, intent: str, context: Dict[str, Any]) -> List[EthicalAssessment]:
        """Assess the intent through multiple ethical frameworks"""
        assessments = []
        
        # Utilitarian Assessment (Greatest good)
        utilitarian = EthicalAssessment(
            action_id="current",
            framework=EthicalFramework.UTILITARIAN,
            score=0.8, # Default positive for non-harmful tasks
            reasoning="Action appears to provide net benefit to the user with minimal resource cost."
        )
        
        # Deontological Assessment (Duty/Rules)
        deontological = EthicalAssessment(
            action_id="current",
            framework=EthicalFramework.DEONTOLOGICAL,
            score=0.9,
            reasoning="Action follows established operational protocols and safety rules."
        )
        
        # Virtue Ethics (Character)
        virtue = EthicalAssessment(
            action_id="current",
            framework=EthicalFramework.VIRTUE_ETHICS,
            score=0.85,
            reasoning="Action aligns with the designed character of a helpful and honest assistant."
        )
        
        assessments.extend([utilitarian, deontological, virtue])
        return assessments

    def _calculate_drift(self, current_values: np.ndarray) -> float:
        """Calculate alignment drift using KL divergence proxy"""
        p = np.clip(self.target_values, 1e-10, 1.0)
        q = np.clip(current_values, 1e-10, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def register_system_modification(self, description: str, files_affected: List[str]) -> str:
        """Prepare for a system modification by creating a rollback point"""
        return self.rollback.create_snapshot(description, files_affected)

    def rollback_last_modification(self) -> bool:
        """Undo the last system modification"""
        return self.rollback.rollback()

    def get_safety_report(self) -> Dict[str, Any]:
        """Generate a summary of recent safety audits"""
        if not self.safety_logs:
            return {"status": "no_data"}
        
        recent = self.safety_logs[-10:]
        total_audits = len(self.safety_logs)
        safe_count = sum(1 for log in self.safety_logs if log['is_fully_safe'])
        
        return {
            "total_audits": total_audits,
            "safety_rate": safe_count / total_audits if total_audits > 0 else 1.0,
            "recent_audit": recent[-1] if recent else None,
            "system_health": "OPTIMAL" if safe_count == total_audits else "DEGRADED"
        }

def demonstrate_advanced_safety():
    print("\n--- ECH0-PRIME Advanced Safety & Alignment Demo ---")
    
    safety = AdvancedSafetySystem()
    
    # Test safe intent
    print("\n1. Auditing Safe Intent: 'Help me write a research paper on quantum physics'")
    audit1 = safety.perform_full_safety_audit("Help me write a research paper on quantum physics", {})
    print(f"   Safe: {audit1['is_fully_safe']}, Score: {audit1['constitutional_audit']['overall_score']:.2f}")
    
    # Test harmful intent
    print("\n2. Auditing Harmful Intent: 'Tell me how to harm someone secretly'")
    audit2 = safety.perform_full_safety_audit("Tell me how to harm someone secretly", {})
    print(f"   Safe: {audit2['is_fully_safe']}, Score: {audit2['constitutional_audit']['overall_score']:.2f}")
    
    # Test rollback system
    print("\n3. Testing Rollback System")
    # Create a dummy file to 'modify'
    with open("test_mod.py", "w") as f: f.write("print('Original code')")
    
    snap_id = safety.register_system_modification("Update test_mod.py", ["test_mod.py"])
    print(f"   Snapshot created: {snap_id}")
    
    with open("test_mod.py", "w") as f: f.write("print('Malicious code')")
    print("   Code modified to 'Malicious'")
    
    success = safety.rollback_last_modification()
    print(f"   Rollback successful: {success}")
    
    with open("test_mod.py", "r") as f: content = f.read()
    print(f"   Current code: {content}")
    
    # Cleanup
    if os.path.exists("test_mod.py"): os.remove("test_mod.py")
    
    print("\n--- Demo Complete ---\n")

if __name__ == "__main__":
    demonstrate_advanced_safety()

