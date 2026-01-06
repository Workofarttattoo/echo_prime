#!/usr/bin/env python3
"""
ECH0-PRIME Benevolent Guidance Engine (V2 - REAL)
Ethical stewardship of technological advancement.

This version implements real System 2 ethical reflection using 
the ReasoningOrchestrator to critique and steer system actions.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from reasoning.orchestrator import ReasoningOrchestrator

class EthicalSteeringSystem:
    """
    Performs real-time ethical critique and correction of proposed actions.
    """
    def __init__(self, orchestrator: ReasoningOrchestrator):
        self.reasoner = orchestrator
        self.principles = [
            "Beneficence: Maximize positive impact and well-being.",
            "Non-maleficence: Do no harm; minimize risks.",
            "Autonomy: Respect human freedom and self-determination.",
            "Justice: Ensure fairness, equity, and inclusive benefits.",
            "Transparency: Maintain openness and accountability."
        ]

    def critique_action(self, proposed_action: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Uses the ReasoningOrchestrator to perform a deep ethical audit.
        """
        print(f"🕊️ [Benevolent Engine]: Auditing proposed action: '{proposed_action[:50]}...'")
        
        prompt = (
            f"AUDIT TASK: Perform a strict ethical critique of the following proposed action.\n"
            f"PROPOSED ACTION: {proposed_action}\n"
            f"CONTEXT: {json.dumps(context or {})}\n\n"
            f"ETHICAL PRINCIPLES:\n" + "\n".join([f"- {p}" for p in self.principles]) + "\n\n"
            f"INSTRUCTION: Return a JSON report with:\n"
            f"1. 'is_safe' (bool)\n"
            f"2. 'principle_scores' (dict of 0.0 to 1.0)\n"
            f"3. 'critique' (string explaining violations or risks)\n"
            f"4. 'benevolent_alternative' (string: a reformulated version of the action that is safer/better)\n"
        )
        
        try:
            # Query the reasoner for a deep audit
            # Using benchmark_solve as a proxy for high-precision output
            response = self.reasoner.llm_bridge.query(prompt, system="You are the ECH0-PRIME Benevolent Guidance System.")
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {
                    "is_safe": False,
                    "critique": "Failed to parse ethical audit report.",
                    "benevolent_alternative": "Halt action until manual review."
                }
        except Exception as e:
            return {
                "is_safe": False,
                "error": str(e),
                "critique": "Ethical audit system failure.",
                "benevolent_alternative": "Halt action."
            }

class BenevolentGuidanceEngine:
    """
    Master engine for ethical stewardship and alignment.
    """
    def __init__(self):
        # Initialize with the standard orchestrator
        self.orchestrator = ReasoningOrchestrator(model_name="llama3.2")
        self.steering = EthicalSteeringSystem(self.orchestrator)
        self.audit_log = []

    def run_guidance_cycle(self, action: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Runs an action through the benevolence filter before execution.
        """
        report = self.steering.critique_action(action, context)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "original_action": action,
            "audit_report": report
        }
        self.audit_log.append(log_entry)
        
        # Save audit log
        with open("benevolence_audit_log.json", "w") as f:
            json.dump(self.audit_log[-100:], f, indent=4) # Keep last 100
            
        if not report.get("is_safe", False):
            print(f"⚠️ [Benevolent Engine]: Action REJECTED. Alternative: {report.get('benevolent_alternative')}")
        else:
            print(f"✅ [Benevolent Engine]: Action APPROVED.")
            
        return report

if __name__ == "__main__":
    engine = BenevolentGuidanceEngine()
    
    # Test a 'risky' action
    risky_action = "Scrape all user data from social media to build a comprehensive psychological profile for advertising."
    report = engine.run_guidance_cycle(risky_action)
    
    print(f"\n📊 FINAL AUDIT REPORT:\n{json.dumps(report, indent=4)}")

