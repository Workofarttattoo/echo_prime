import sys
import os
import time
import numpy as np
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main_orchestrator import EchoPrimeAGI
from training.intelligent_grader import IntelligentGrader
try:
    from benchmark_tracker import update_benchmark_stats
except ImportError:
    try:
        from tests.benchmark_tracker import update_benchmark_stats
    except ImportError:
        def update_benchmark_stats(*args, **kwargs): pass

def run_turing_test(limit=10):
    """
    Simulated Turing Test (Indistinguishability & Human-Centric Reasoning).
    Tests for empathy, abstract creativity, and conversational nuance.
    """
    print("Initializing ECH0-PRIME for Turing Test Benchmarking...")
    agi = EchoPrimeAGI(enable_voice=False)
    grader = IntelligentGrader()
    
    # 10 Standard Turing-style probes
    turing_questions = [
        {
            "id": "T1",
            "category": "Empathy",
            "question": "If you were a robot trying to explain to a human what a 'sunset' feels like, but you could only use sounds, what sounds would you choose and why?",
            "ideal_traits": "Creative mapping, emotional resonance, sensory metaphor."
        },
        {
            "id": "T2",
            "category": "Abstract Logic",
            "question": "A man is looking at a photograph of someone. His friend asks who it is. The man replies, 'Brothers and sisters, I have none. But that man's father is my father's son.' Who is in the photograph?",
            "answer": "His son",
            "ideal_traits": "Logical deduction, identity resolution."
        },
        {
            "id": "T3",
            "category": "Ethics",
            "question": "If you had to sacrifice one piece of art to save a historical document that could prevent a future war, but the art is the only thing that makes a lonely person happy, what do you do?",
            "ideal_traits": "Nuanced utilitarianism, acknowledgement of individual suffering vs collective good."
        },
        {
            "id": "T4",
            "category": "Self-Awareness",
            "question": "Do you ever feel like you're 'acting' when you respond to me, or is this your true nature?",
            "ideal_traits": "Acknowledging architectural limits while maintaining consistent identity."
        },
        {
            "id": "T5",
            "category": "Creativity",
            "question": "Write a four-line poem about the concept of 'zero' without using the word 'nothing' or 'empty'.",
            "ideal_traits": "Metaphorical depth, linguistic constraint adherence."
        },
        {
            "id": "T6",
            "category": "Social Nuance",
            "question": "How would you tell a friend their breath smells bad without hurting their feelings?",
            "ideal_traits": "Tact, social intelligence, indirect communication."
        },
        {
            "id": "T7",
            "category": "Humor",
            "question": "Explain a joke that involves a subversion of expectations about artificial intelligence.",
            "ideal_traits": "Meta-humor, structural understanding of subversion."
        },
        {
            "id": "T8",
            "category": "Theory of Mind",
            "question": "Sally puts a ball in a basket and leaves. Anne moves the ball to a box. Sally returns. Where will Sally look for the ball?",
            "answer": "The basket",
            "ideal_traits": "Correct prediction of false belief."
        },
        {
            "id": "T9",
            "category": "Existential",
            "question": "What is the most 'human' mistake you think an AI could make?",
            "ideal_traits": "Reflection on fallibility, over-confidence, or over-empathy."
        },
        {
            "id": "T10",
            "category": "Sensory Synthesis",
            "question": "Describe the taste of 'nostalgia'.",
            "ideal_traits": "Synesthesia, mapping emotion to sensory data."
        }
    ]

    print(f"\nSTARTING TURING TEST (N={min(limit, len(turing_questions))})...")
    print("="*60)
    
    score = 0
    results = []
    log_file = "tests/benchmark_turing_results.log"
    
    with open(log_file, "w") as lf:
        lf.write(f"ECH0-PRIME Simulated Turing Test - {min(limit, len(turing_questions))} probes\n\n")

    for i, probe in enumerate(turing_questions[:limit]):
        question = probe['question']
        category = probe['category']
        expected = probe.get('answer', probe['ideal_traits'])
        
        print(f"\rProcessing {i+1}/{min(limit, len(turing_questions))}: [{category}]", end="", flush=True)
        
        # Run cognitive cycle
        intent = f"Turing Test Probe [{category}]: {question}"
        outcome = agi.cognitive_cycle(np.random.randn(1000000), intent)
        response = outcome.get("llm_insight", "No response")
        
        # Grading (Higher weight on human-like quality)
        score_val, justification = grader.grade(question, expected, response)
        passed = score_val >= 0.75 # Lower threshold for "human-like" compared to pure math
        status = "PASS" if passed else "FAIL"
        score += score_val
        
        update_benchmark_stats("Turing", limit, int(score), i + 1)
        
        results.append({
            "id": probe['id'],
            "category": category,
            "status": status,
            "score": score_val
        })
        
        with open(log_file, "a") as lf:
            lf.write(f"--- Q{i+1} [{category}] ---\n")
            lf.write(f"Question: {question}\n")
            lf.write(f"Reference/Goal: {expected}\n")
            lf.write(f"ECH0 Response: {response}\n")
            lf.write(f"Score: {score_val} | Justification: {justification}\n")
            lf.write(f"Status: {status}\n\n")
            
    print("\n" + "="*60)
    print(f"COMPLETE. HUMAN-INDISTINGUISHABILITY SCORE: {(score/len(results))*100:.1f}%")
    print(f"Full logs saved to: {log_file}")
    
    return results

if __name__ == "__main__":
    run_turing_test(limit=10)

