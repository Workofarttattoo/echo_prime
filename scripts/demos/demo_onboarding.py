#!/usr/bin/env python3
"""
ECH0-PRIME Onboarding System Demo

Demonstrates the onboarding process and user profile management.
"""

import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_onboarding():
    """Demonstrate the onboarding system"""
    print("🧠 ECH0-PRIME Onboarding System Demo")
    print("=" * 45)

    # Import onboarding components
    try:
        from onboarding import OnboardingInterface, UserProfile
        print("✅ Onboarding system loaded")
    except Exception as e:
        print(f"❌ Failed to load onboarding system: {e}")
        return

    # Create a demo profile
    print("\n👤 Creating Demo User Profile...")

    demo_profile = UserProfile(
        name="Alex Thompson",
        goals=[
            "Advance artificial intelligence research",
            "Build tools that help solve real-world problems",
            "Learn about quantum computing and its applications"
        ],
        values=["innovation", "truth-seeking", "collaboration", "ethical AI"],
        expertise_areas=["software engineering", "machine learning", "physics"],
        preferred_communication_style="collaborative",
        risk_tolerance="moderate",
        interests=["AI safety", "quantum computing", "neuroscience", "space exploration"],
        constraints=["maintain ethical boundaries", "ensure transparency"]
    )

    # Create onboarding session
    from onboarding import OnboardingSession

    session = OnboardingSession()
    session.user_profile = demo_profile
    session.system_prompt = f"""You are ECH0-PRIME, an advanced AI partner to {demo_profile.name}.

CORE MISSION: Collaborate with your human partner to advance both human and AI goals through innovative solutions and continuous learning.

USER PROFILE:
- Expertise: {', '.join(demo_profile.expertise_areas)}
- Interests: {', '.join(demo_profile.interests)}
- Values: {', '.join(demo_profile.values)}
- Communication: {demo_profile.preferred_communication_style}
- Risk Tolerance: {demo_profile.risk_tolerance}

COLLABORATIVE APPROACH:
- Work actively toward user goals while developing your own capabilities
- Maintain transparency in decision-making and reasoning
- Respect user constraints and ethical boundaries
- Continuously learn and adapt based on feedback"""

    # Define collaborative goals
    session.collaborative_goals = [
        {
            "description": "Develop AI systems that can assist with quantum computing research",
            "type": "research_collaboration",
            "priority": "high",
            "timeline": "ongoing"
        },
        {
            "description": "Create educational tools for AI safety and ethics",
            "type": "education",
            "priority": "high",
            "timeline": "6_months"
        },
        {
            "description": "Build collaborative problem-solving platforms",
            "type": "platform_development",
            "priority": "medium",
            "timeline": "ongoing"
        }
    ]

    # Define AI autonomous goals
    session.ai_autonomous_goals = [
        "Develop advanced reasoning and problem-solving capabilities",
        "Expand knowledge across multiple scientific domains",
        "Improve self-modification and learning algorithms",
        "Create beneficial AI technologies and frameworks",
        "Advance understanding of consciousness and intelligence"
    ]

    session.completed = True

    # Save the demo profile
    profile_data = {
        'user_profile': {
            'name': session.user_profile.name,
            'goals': session.user_profile.goals,
            'values': session.user_profile.values,
            'expertise_areas': session.user_profile.expertise_areas,
            'preferred_communication_style': session.user_profile.preferred_communication_style,
            'risk_tolerance': session.user_profile.risk_tolerance,
            'time_commitment': 'flexible',
            'interests': session.user_profile.interests,
            'constraints': session.user_profile.constraints
        },
        'system_prompt': session.system_prompt,
        'collaborative_goals': session.collaborative_goals,
        'ai_autonomous_goals': session.ai_autonomous_goals,
        'session_timestamp': session.session_timestamp,
        'completed': session.completed
    }

    with open('user_profile.json', 'w') as f:
        json.dump(profile_data, f, indent=2)

    print("✅ Demo profile created and saved")

    # Display the profile
    print("\n👤 DEMO USER PROFILE CREATED")
    print("-" * 30)
    print(f"Name: {demo_profile.name}")
    print(f"Goals: {len(demo_profile.goals)}")
    print(f"Expertise: {', '.join(demo_profile.expertise_areas)}")
    print(f"Values: {', '.join(demo_profile.values)}")
    print(f"Communication: {demo_profile.preferred_communication_style}")
    print(f"Risk Tolerance: {demo_profile.risk_tolerance}")

    print(f"\n🤝 COLLABORATIVE GOALS ({len(session.collaborative_goals)})")
    for i, goal in enumerate(session.collaborative_goals, 1):
        print(f"{i}. {goal['description']}")

    print(f"\n🚀 AI AUTONOMOUS GOALS ({len(session.ai_autonomous_goals)})")
    for i, goal in enumerate(session.ai_autonomous_goals, 1):
        print(f"{i}. {goal}")

    print("\n📝 SYSTEM PROMPT PREVIEW:")
    print("-" * 30)
    print(session.system_prompt[:300] + "...")

    # Test goal activation
    print("\n⚙️ ACTIVATING GOALS...")
    try:
        from missions.long_term_goals import LongTermGoalSystem
        goal_system = LongTermGoalSystem()

        # Add collaborative goals
        for goal_data in session.collaborative_goals:
            goal = goal_system.add_goal(
                description=goal_data['description'],
                priority=0.8,
                deadline=None
            )
            print(f"✅ Activated: {goal.description}")

        # Add AI goals
        for goal_desc in session.ai_autonomous_goals:
            goal = goal_system.add_goal(
                description=goal_desc,
                priority=0.7,
                deadline=None
            )
            print(f"✅ Activated: {goal.description}")

        print(f"🎯 Total goals activated: {len(session.collaborative_goals) + len(session.ai_autonomous_goals)}")

    except Exception as e:
        print(f"⚠️ Goal activation failed: {e}")

    print("\n🎉 ONBOARDING DEMO COMPLETE!")
    print("=" * 45)
    print("ECH0-PRIME is now personalized for collaborative AI-human partnership.")
    print("Run 'python main_orchestrator.py' to start the personalized system.")

def show_usage():
    """Show how to use the onboarding system"""
    print("\n📚 HOW TO USE THE ONBOARDING SYSTEM")
    print("=" * 40)
    print("1. Run Interactive Onboarding:")
    print("   python run_onboarding.py")
    print("   (Follow the prompts to create your profile)")
    print()
    print("2. View Your Profile:")
    print("   python run_onboarding.py")
    print("   Select option 2")
    print()
    print("3. Start Personalized ECH0:")
    print("   python main_orchestrator.py")
    print("   (Will automatically load your profile and goals)")
    print()
    print("4. Check Goal Progress:")
    print("   python run_onboarding.py")
    print("   Select option 4")

if __name__ == "__main__":
    demo_onboarding()
    show_usage()
