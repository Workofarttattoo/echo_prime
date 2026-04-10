#!/usr/bin/env python3
"""
ECH0-PRIME Feedback Learning Demo
Demonstrates how the system learns from user feedback and continuously improves.
"""

import asyncio
import time
import json
from main_orchestrator import EchoPrimeAGI
from feedback_loop import FeedbackType, FeedbackPriority


async def demo_feedback_learning():
    """Demonstrate the feedback learning system"""
    print("🧠 ECH0-PRIME Feedback Learning Demo")
    print("=" * 50)

    # Initialize the AGI system
    print("Initializing ECH0-PRIME...")
    agi = EchoPrimeAGI(enable_voice=False)  # Disable voice for demo

    # Wait a moment for initialization
    await asyncio.sleep(2)

    print("\n📚 Teaching the system through feedback...")
    print("-" * 40)

    # Example 1: Performance feedback
    print("\n1. Submitting performance feedback...")
    await agi.submit_feedback(
        FeedbackType.PERFORMANCE_METRIC,
        {
            'metric_name': 'response_accuracy',
            'value': 0.75,
            'context': 'general_question_answering',
            'improvement_needed': 'more_detailed_responses'
        },
        source="user_evaluation",
        priority=FeedbackPriority.HIGH
    )

    # Example 2: User correction
    print("2. Submitting user correction...")
    await agi.submit_feedback(
        FeedbackType.USER_CORRECTION,
        {
            'original_response': 'The answer is 42',
            'correction': 'Please provide more context and explanation',
            'reason': 'response_too_brief'
        },
        source="user_direct",
        priority=FeedbackPriority.CRITICAL
    )

    # Example 3: Error report
    print("3. Submitting error feedback...")
    await agi.submit_feedback(
        FeedbackType.ERROR_REPORT,
        {
            'error_type': 'ValueError',
            'error_message': 'Invalid input format',
            'context': 'data_processing',
            'frequency': 'occasional'
        },
        source="system_monitor",
        priority=FeedbackPriority.MEDIUM
    )

    # Example 4: Success indicator
    print("4. Submitting success feedback...")
    await agi.submit_feedback(
        FeedbackType.SUCCESS_INDICATOR,
        {
            'task_type': 'problem_solving',
            'success_rate': 0.9,
            'context': 'mathematical_reasoning',
            'user_satisfaction': 0.95
        },
        source="task_completion",
        priority=FeedbackPriority.MEDIUM
    )

    # Example 5: User preference
    print("5. Submitting user preference...")
    await agi.submit_feedback(
        FeedbackType.USER_PREFERENCE,
        {
            'preference_type': 'communication_style',
            'preferred_style': 'concise_and_clear',
            'avoided_style': 'verbose_and_wordy',
            'examples': ['Keep explanations focused', 'Use bullet points when appropriate']
        },
        source="user_profile",
        priority=FeedbackPriority.HIGH
    )

    print("\n⏳ Waiting for learning cycle to process feedback...")
    await asyncio.sleep(5)  # Let the learning cycle run

    # Force a learning cycle
    print("🔄 Forcing learning cycle...")
    await agi.feedback_loop.force_learning_cycle()

    # Wait for processing
    await asyncio.sleep(3)

    # Show learning statistics
    print("\n📊 Learning Statistics:")
    print("-" * 30)
    stats = agi.get_learning_stats()
    print(json.dumps(stats, indent=2))

    # Demonstrate improved behavior
    print("\n🎯 Testing learned improvements...")
    print("-" * 35)

    # Test 1: Check if the system adapted its response style
    test_input = np.random.randn(10000)  # Mock sensory input
    result = agi.cognitive_cycle(test_input, "Explain quantum computing briefly")

    if 'llm_insight' in result:
        response = result['llm_insight']
        # Check if response is more concise (learned from feedback)
        if len(response.split()) < 100:  # Reasonable length check
            print("✅ Response style improved: More concise")
        else:
            print("⚠️ Response style may need more learning")

    # Test 2: Check error handling improvements
    try:
        # This might trigger improved error handling
        agi.cognitive_cycle(None, "Process invalid input")
        print("✅ Error handling working properly")
    except Exception as e:
        print(f"ℹ️ Error handling learning in progress: {e}")

    # Show final learning stats
    print("\n🏁 Final Learning Summary:")
    print("-" * 28)
    final_stats = agi.get_learning_stats()

    improvements = final_stats.get('adaptation_stats', {}).get('successful_adaptations', 0)
    total_feedback = final_stats.get('feedback_stats', {}).get('total_feedback', 0)
    processed_feedback = final_stats.get('feedback_stats', {}).get('processed_feedback', 0)

    print(f"Total feedback received: {total_feedback}")
    print(f"Feedback processed: {processed_feedback}")
    print(f"Successful adaptations: {improvements}")
    print(".1f"    if improvements > 0:
        print("🎉 System successfully learned and adapted from feedback!")
    else:
        print("📚 System is still learning from the feedback...")

    # Cleanup
    agi.feedback_loop.stop_learning_loop()
    print("\n✨ Demo completed!")


async def interactive_feedback_session():
    """Interactive session where users can teach the system"""
    print("🎓 ECH0-PRIME Interactive Learning Session")
    print("=" * 45)
    print("Teach the system by providing feedback on its responses.")
    print("Type 'quit' to end the session.\n")

    agi = EchoPrimeAGI(enable_voice=False)

    await asyncio.sleep(2)  # Let system initialize

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Process user input through cognitive system
        print("🤔 ECH0-PRIME is thinking...")
        sensory_input = np.random.randn(10000)  # Mock sensory input
        result = agi.cognitive_cycle(sensory_input, user_input)

        # Show system response
        if 'llm_insight' in result:
            print(f"ECH0-PRIME: {result['llm_insight'][:200]}...")
        else:
            print("ECH0-PRIME: [Processing incomplete]")

        # Ask for feedback
        feedback = input("\nFeedback (good/bad/correct/improve/enter to skip): ").strip().lower()

        if feedback == 'good':
            await agi.submit_feedback(
                FeedbackType.SUCCESS_INDICATOR,
                {'quality': 'good', 'context': user_input},
                source="interactive_user",
                priority=FeedbackPriority.MEDIUM
            )
            print("👍 Feedback recorded: Response was good")

        elif feedback == 'bad':
            await agi.submit_feedback(
                FeedbackType.PERFORMANCE_METRIC,
                {'metric_name': 'response_quality', 'value': 0.3, 'context': user_input},
                source="interactive_user",
                priority=FeedbackPriority.HIGH
            )
            print("👎 Feedback recorded: Response needs improvement")

        elif feedback == 'correct':
            correction = input("What should it have said? ")
            await agi.submit_feedback(
                FeedbackType.USER_CORRECTION,
                {'original_context': user_input, 'correction': correction},
                source="interactive_user",
                priority=FeedbackPriority.CRITICAL
            )
            print("✅ Correction recorded")

        elif feedback.startswith('improve'):
            improvement = input("How should it improve? ")
            await agi.submit_feedback(
                FeedbackType.USER_PREFERENCE,
                {'improvement_request': improvement, 'context': user_input},
                source="interactive_user",
                priority=FeedbackPriority.HIGH
            )
            print("🔧 Improvement request recorded")

        elif feedback:
            # Generic feedback
            await agi.submit_feedback(
                FeedbackType.ENVIRONMENTAL_FEEDBACK,
                {'feedback': feedback, 'context': user_input},
                source="interactive_user",
                priority=FeedbackPriority.MEDIUM
            )
            print("📝 Feedback recorded")

        # Periodic learning
        if agi.feedback_loop.learning_cycle_count % 5 == 0:
            print("🔄 Running learning cycle...")
            await agi.feedback_loop.force_learning_cycle()

        print()

    # Show final stats
    print("\n📊 Session Summary:")
    stats = agi.get_learning_stats()
    print(f"Learning cycles completed: {stats.get('total_cycles', 0)}")
    print(f"Feedback items processed: {stats.get('feedback_stats', {}).get('total_feedback', 0)}")
    print(f"Adaptations made: {stats.get('adaptation_stats', {}).get('successful_adaptations', 0)}")

    agi.feedback_loop.stop_learning_loop()


if __name__ == "__main__":
    import numpy as np

    print("ECH0-PRIME Feedback Learning Demo")
    print("1. Automated demo")
    print("2. Interactive learning session")

    choice = input("Choose demo type (1 or 2): ").strip()

    if choice == "2":
        asyncio.run(interactive_feedback_session())
    else:
        asyncio.run(demo_feedback_learning())


