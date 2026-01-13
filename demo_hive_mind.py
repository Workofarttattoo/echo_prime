#!/usr/bin/env python3
"""
ECH0-PRIME Hive Mind Demonstration
Shows the collective intelligence system solving real problems.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from missions.hive_mind import HiveMindOrchestrator

def demonstrate_algorithm_design():
    """Demonstrate hive mind designing an algorithm"""
    print("\n🧮 DEMONSTRATION: Algorithm Design")
    print("=" * 50)

    hive = HiveMindOrchestrator(num_nodes=4)

    # Submit algorithm design task
    task_id = hive.submit_task(
        "Design an efficient algorithm for finding the k-th largest element in an unsorted array",
        domain="engineering"
    )

    print(f"🎯 Task submitted: {task_id}")

    # Run the hive cycle
    result = hive.run_hive_cycle()

    # Display results
    if result['completed_tasks']:
        task = result['completed_tasks'][0]
        print("\n📊 SOLUTION RESULTS:")
        print(f"Confidence: {task['solution']['confidence']:.2f}")
        print(f"Method: {task['solution']['emergence_method']}")
        print(f"Subtasks completed: {task['subtasks_completed']}")

        # Show the actual solution
        solution = task['solution']['solution']
        print("\n💡 SOLUTION:")
        print(solution)

    hive.shutdown_hive()
    return result

def demonstrate_optimization_problem():
    """Demonstrate hive mind optimizing a complex system"""
    print("\n⚡ DEMONSTRATION: System Optimization")
    print("=" * 50)

    hive = HiveMindOrchestrator(num_nodes=5)

    # Submit optimization task
    task_id = hive.submit_task(
        "Optimize a distributed database system for high-throughput read/write operations under varying load conditions",
        domain="engineering"
    )

    print(f"🎯 Task submitted: {task_id}")

    # Run multiple cycles for complex problem
    total_results = []
    for cycle in range(2):
        print(f"\n🔄 Running hive cycle {cycle + 1}...")
        result = hive.run_hive_cycle()
        total_results.extend(result.get('completed_tasks', []))

    # Display results
    if total_results:
        print("\n📊 OPTIMIZATION RESULTS:")
        for i, task in enumerate(total_results):
            print(f"\nTask {i+1}:")
            print(f"  Confidence: {task['solution']['confidence']:.2f}")
            print(f"  Subtasks: {task['subtasks_completed']}")
            print(f"  Solution: {task['solution']['solution'][:100]}...")

    hive.shutdown_hive()
    return total_results

def demonstrate_scientific_discovery():
    """Demonstrate hive mind conducting scientific discovery"""
    print("\n🔬 DEMONSTRATION: Scientific Discovery")
    print("=" * 50)

    hive = HiveMindOrchestrator(num_nodes=3)

    # Submit scientific discovery task
    task_id = hive.submit_task(
        "Investigate the relationship between neural network architecture complexity and emergent generalization capabilities",
        domain="research"
    )

    print(f"🎯 Task submitted: {task_id}")

    # Run discovery process
    result = hive.run_hive_cycle()

    # Display results
    if result['completed_tasks']:
        task = result['completed_tasks'][0]
        print("\n📊 DISCOVERY RESULTS:")
        print(f"Confidence: {task['solution']['confidence']:.2f}")
        print(f"Research approach: {task['solution']['emergence_method']}")

        # Show hypotheses generated
        solution = task['solution']['solution']
        print("\n🔍 HYPOTHESES GENERATED:")
        print(solution)

    hive.shutdown_hive()
    return result

def demonstrate_creative_problem():
    """Demonstrate hive mind solving a creative problem"""
    print("\n🎨 DEMONSTRATION: Creative Problem Solving")
    print("=" * 50)

    hive = HiveMindOrchestrator(num_nodes=4)

    # Submit creative task
    task_id = hive.submit_task(
        "Design an innovative user interface paradigm that combines augmented reality, brain-computer interfaces, and adaptive personalization",
        domain="innovation"
    )

    print(f"🎯 Task submitted: {task_id}")

    # Run creative process
    result = hive.run_hive_cycle()

    # Display results
    if result['completed_tasks']:
        task = result['completed_tasks'][0]
        print("\n📊 CREATIVE RESULTS:")
        print(f"Innovation confidence: {task['solution']['confidence']:.2f}")
        print(f"Creative approach: {task['solution']['emergence_method']}")

        # Show creative solution
        solution = task['solution']['solution']
        print("\n💡 INNOVATIVE SOLUTION:")
        print(solution)

    hive.shutdown_hive()
    return result


def demonstrate_stem_cell_regrowth():
    """Demonstrate hive mind solving stem cell regrowth for major body parts"""
    print("\n🧬 DEMONSTRATION: Stem Cell Regrowth Optimization")
    print("=" * 50)

    hive = HiveMindOrchestrator(num_nodes=6)

    body_parts = [
        "brain",
        "heart",
        "liver",
        "kidney",
        "lungs",
        "skin",
        "muscle",
        "bone"
    ]

    task_ids = []
    for part in body_parts:
        task_desc = f"Optimize stem cell regrowth process for the human {part}"
        task_id = hive.submit_task(task_desc, domain="biology", complexity=2.0)
        task_ids.append(task_id)
        print(f"🎯 Task submitted for {part}: {task_id}")

    # Run cycles indefinitely, tailing stats
    cycles = 0
    while True:
        result = hive.run_hive_cycle()
        completed = len(result.get('completed_tasks', []))
        if completed:
            print(f"🔄 Cycle {cycles+1}: Completed {completed} stem cell tasks")
        cycles += 1
        if all(hive.tasks[tid].status == 'completed' for tid in task_ids):
            break
        time.sleep(1)

    # Summarize results
    print("\n📊 STEM CELL REGROWTH RESULTS:")
    for tid in task_ids:
        task = hive.tasks[tid]
        sol = task.consensus_solution
        print(f"- {task.description}: confidence {sol.get('confidence',0):.2f}")

    hive.shutdown_hive()
    return result

def run_full_demonstration():
    """Run complete hive mind demonstration suite"""
    print("🧠 ECH0-PRIME HIVE MIND DEMONSTRATION SUITE")
    print("=" * 60)
    print("Demonstrating collective intelligence across multiple domains...")
    print()

    demonstrations = [
        ("Algorithm Design", demonstrate_algorithm_design),
        ("System Optimization", demonstrate_optimization_problem),
        ("Scientific Discovery", demonstrate_scientific_discovery),
        ("Creative Innovation", demonstrate_creative_problem),
        ("Stem Cell Regrowth", demonstrate_stem_cell_regrowth)
    ]

    results_summary = []

    for demo_name, demo_func in demonstrations:
        try:
            print(f"\n🚀 STARTING: {demo_name}")
            result = demo_func()
            results_summary.append((demo_name, "SUCCESS", len(result.get('completed_tasks', []))))
            print(f"✅ COMPLETED: {demo_name}")
        except Exception as e:
            print(f"❌ FAILED: {demo_name} - {e}")
            results_summary.append((demo_name, "FAILED", 0))

    # Final summary
    print("\n" + "=" * 60)
    print("🎯 DEMONSTRATION SUMMARY")
    print("=" * 60)

    total_tasks = 0
    successful_demos = 0

    for demo_name, status, tasks in results_summary:
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} {demo_name}: {status} ({tasks} tasks completed)")
        if status == "SUCCESS":
            successful_demos += 1
            total_tasks += tasks

    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"• Demonstrations: {successful_demos}/{len(demonstrations)} successful")
    print(f"• Total tasks completed: {total_tasks}")
    print(f"• Success rate: {(successful_demos/len(demonstrations))*100:.1f}%")

    if successful_demos == len(demonstrations):
        print("\n🎉 HIVE MIND DEMONSTRATION: COMPLETE SUCCESS!")
        print("The collective intelligence system is fully operational.")
        print("Ready for real-world problem solving across all domains.")
    else:
        print(f"\n⚠️ HIVE MIND DEMONSTRATION: {successful_demos} successes, {len(demonstrations)-successful_demos} issues detected.")

if __name__ == "__main__":
    run_full_demonstration()
