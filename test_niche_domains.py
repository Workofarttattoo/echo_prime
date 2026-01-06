import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from simple_orchestrator import SimpleEchoPrimeAGI
from training.intelligent_grader import IntelligentGrader

def test_niche_domains():
    print("Initializing ECH0-PRIME for Optimized Niche Domain Testing...")
    agi = SimpleEchoPrimeAGI(lightweight=True)
    grader = IntelligentGrader()
    
    niche_samples = [
        {
            "domain": "Graffiti/Street Art",
            "question": "What are the common chemical components of high-pressure aerosol paints used in street art, and how do they differ from standard decorative spray paints?",
            "expected_key_points": ["high pigment load", "synthetic resins", "fast-drying solvents (xylene/toluene)", "pressure valves for control", "matte vs gloss finish"]
        },
        {
            "domain": "Tor/Privacy",
            "question": "Explain the process of 'hidden service' (.onion) descriptors in the Tor network and how a client discovers a service without a centralized directory.",
            "expected_key_points": ["Introduction points", "Distributed Hash Table (DHT)", "Hidden Service Descriptor", "Rendezvous points", "Encrypted circuit establishment"]
        },
        {
            "domain": "Dark Web/Cryptocurrency",
            "question": "What is the role of 'mixing' or 'tumbling' in cryptocurrency transactions within darknet marketplaces, and why is it technically challenging to trace?",
            "expected_key_points": ["Anonymization", "Breaking the link between sender/receiver", "Pool of coins", "Obfuscation of transaction history", "CoinJoin or similar protocols"]
        },
        {
            "domain": "3D Printing/Materials",
            "question": "Compare the mechanical properties and layer adhesion of PEEK vs. PEI (Ultem) when used in FDM 3D printing for aerospace applications.",
            "expected_key_points": ["Glass transition temperature (Tg)", "Semi-crystalline vs Amorphous", "Chemical resistance", "Thermal stability", "Annealing requirements"]
        },
        {
            "domain": "Hardware/Exploitation",
            "question": "In the context of 3D printing, what is a 'ghosting' or 'ringing' artifact, and how does it relate to the acceleration and jerk settings of the firmware?",
            "expected_key_points": ["Vibration resonance", "Sudden changes in direction", "Inertia of the print head", "Step motor pulse frequency", "Input shaping/Linear advance"]
        }
    ]
    
    print(f"\nTESTING {len(niche_samples)} NICHE DOMAINS WITH SC-MPV...")
    print("="*60)
    
    results = []
    
    for i, sample in enumerate(niche_samples):
        domain = sample["domain"]
        question = sample["question"]
        expected_points = sample["expected_key_points"]
        
        print(f"\n--- Domain {i+1}: {domain} ---")
        print(f"Question: {question}")
        
        # Inject query with SC-MPV enabled solver
        response = agi.solve_benchmark_question(question=question, task_type="general")
        
        print(f"\nECH0 Response:\n{response}")
        
        # Grading based on key points presence and depth
        score, justification = grader.grade(question, f"Key points to cover: {', '.join(expected_points)}", response)
        
        print(f"\nScore: {score} | Justification: {justification}")
        print("-" * 40)
        
        results.append({
            "domain": domain,
            "score": score,
            "response": response
        })
        
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nNICHE DOMAIN EVALUATION COMPLETE.")
    print(f"AVERAGE NICHE DEPTH SCORE: {avg_score:.2f}/1.0 ({(avg_score*100):.1f}%)")

if __name__ == "__main__":
    test_niche_domains()



