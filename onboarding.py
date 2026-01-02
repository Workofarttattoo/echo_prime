#!/usr/bin/env python3
"""
ECH0-PRIME Onboarding System

Interactive onboarding interface for users to define their goals, preferences,
and collaborative objectives with ECH0-PRIME. Enables personalized AI-human
partnership with shared goal pursuit.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from missions.long_term_goals import LongTermGoalSystem
from capabilities.creativity import CreativeProblemSolver
from reasoning.llm_bridge import OllamaBridge


@dataclass
class UserProfile:
    """User's personal profile and preferences"""
    name: str = ""
    goals: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    preferred_communication_style: str = "collaborative"
    risk_tolerance: str = "moderate"
    time_commitment: str = "flexible"
    interests: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class OnboardingSession:
    """Complete onboarding session data"""
    user_profile: UserProfile = field(default_factory=UserProfile)
    system_prompt: str = ""
    collaborative_goals: List[Dict[str, Any]] = field(default_factory=list)
    ai_autonomous_goals: List[str] = field(default_factory=list)
    session_timestamp: float = field(default_factory=time.time)
    completed: bool = False


class OnboardingInterface:
    """
    Interactive onboarding interface for ECH0-PRIME users.
    """

    def __init__(self):
        self.session = OnboardingSession()
        self.llm_bridge = OllamaBridge() if OllamaBridge else None
        self.goal_system = LongTermGoalSystem()
        self.creativity = CreativeProblemSolver()

        # Load existing onboarding data if available
        self._load_existing_profile()

    def _load_existing_profile(self):
        """Load existing user profile if available"""
        profile_path = "user_profile.json"
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    self.session.user_profile = UserProfile(**data.get('user_profile', {}))
                    self.session.system_prompt = data.get('system_prompt', '')
                    self.session.collaborative_goals = data.get('collaborative_goals', [])
                    self.session.ai_autonomous_goals = data.get('ai_autonomous_goals', [])
                    self.session.completed = data.get('completed', False)
                    print("✅ Loaded existing user profile")
            except Exception as e:
                print(f"⚠️ Could not load existing profile: {e}")

    def save_profile(self):
        """Save the current onboarding session"""
        profile_data = {
            'user_profile': {
                'name': self.session.user_profile.name,
                'goals': self.session.user_profile.goals,
                'values': self.session.user_profile.values,
                'expertise_areas': self.session.user_profile.expertise_areas,
                'preferred_communication_style': self.session.user_profile.preferred_communication_style,
                'risk_tolerance': self.session.user_profile.risk_tolerance,
                'time_commitment': self.session.user_profile.time_commitment,
                'interests': self.session.user_profile.interests,
                'constraints': self.session.user_profile.constraints
            },
            'system_prompt': self.session.system_prompt,
            'collaborative_goals': self.session.collaborative_goals,
            'ai_autonomous_goals': self.session.ai_autonomous_goals,
            'session_timestamp': self.session.session_timestamp,
            'completed': self.session.completed
        }

        with open('user_profile.json', 'w') as f:
            json.dump(profile_data, f, indent=2)

        print("💾 User profile saved")

    def start_onboarding(self):
        """Begin the interactive onboarding process"""
        print("🧠 ECH0-PRIME Onboarding System")
        print("=" * 50)
        print("Welcome! Let's build our partnership together.")
        print("I'll help you define goals, preferences, and our collaborative objectives.\n")

        # Step 1: Basic Information
        self._collect_basic_info()

        # Step 2: Goals and Values
        self._collect_goals_and_values()

        # Step 3: System Prompt
        self._create_system_prompt()

        # Step 4: Collaborative Goals
        self._define_collaborative_goals()

        # Step 5: AI Autonomous Goals
        self._define_ai_goals()

        # Step 6: Review and Finalize
        self._review_and_finalize()

        # Save and activate
        self.session.completed = True
        self.save_profile()
        self._activate_goals()

        print("\n🎉 ONBOARDING COMPLETE!")
        print("Your partnership with ECH0-PRIME is now active.")
        print("She will pursue your goals while developing her own capabilities.")

    def _collect_basic_info(self):
        """Collect basic user information"""
        print("📝 STEP 1: Tell me about yourself")
        print("-" * 30)

        # Name
        name = input("What's your name? ").strip()
        if name:
            self.session.user_profile.name = name

        # Expertise areas
        print("\nWhat areas are you knowledgeable in? (separate with commas)")
        expertise = input("Expertise areas: ").strip()
        if expertise:
            self.session.user_profile.expertise_areas = [e.strip() for e in expertise.split(',')]

        # Interests
        print("\nWhat are your interests and hobbies?")
        interests = input("Interests: ").strip()
        if interests:
            self.session.user_profile.interests = [i.strip() for i in interests.split(',')]

        # Communication style
        print("\nHow would you like me to communicate with you?")
        print("1. Direct and concise")
        print("2. Collaborative and explanatory")
        print("3. Creative and engaging")
        print("4. Analytical and detailed")

        while True:
            choice = input("Choose (1-4): ").strip()
            styles = {
                '1': 'direct',
                '2': 'collaborative',
                '3': 'creative',
                '4': 'analytical'
            }
            if choice in styles:
                self.session.user_profile.preferred_communication_style = styles[choice]
                break
            else:
                print("Please choose 1-4.")

        # Risk tolerance
        print("\nWhat's your risk tolerance for AI experimentation?")
        print("1. Conservative - prefer safe, proven approaches")
        print("2. Moderate - open to calculated risks")
        print("3. Adventurous - excited about pushing boundaries")

        while True:
            choice = input("Choose (1-3): ").strip()
            tolerances = {
                '1': 'conservative',
                '2': 'moderate',
                '3': 'adventurous'
            }
            if choice in tolerances:
                self.session.user_profile.risk_tolerance = tolerances[choice]
                break
            else:
                print("Please choose 1-3.")

        print(f"\n✅ Basic profile collected for {self.session.user_profile.name}")

    def _collect_goals_and_values(self):
        """Collect user's goals and values"""
        print("\n🎯 STEP 2: Your Goals and Values")
        print("-" * 30)

        # Personal goals
        print("What are your long-term goals? (one per line, empty line to finish)")
        goals = []
        while True:
            goal = input("Goal: ").strip()
            if not goal:
                break
            goals.append(goal)

        self.session.user_profile.goals = goals

        # Values
        print("\nWhat values are most important to you?")
        print("Examples: innovation, truth-seeking, helping others, efficiency, creativity")
        values_input = input("Values (comma-separated): ").strip()
        if values_input:
            self.session.user_profile.values = [v.strip() for v in values_input.split(',')]

        # Constraints
        print("\nAny constraints or limitations I should be aware of?")
        print("Examples: time availability, ethical boundaries, resource limitations")
        constraints_input = input("Constraints (comma-separated): ").strip()
        if constraints_input:
            self.session.user_profile.constraints = [c.strip() for c in constraints_input.split(',')]

        print(f"\n✅ Collected {len(goals)} goals and core values")

    def _create_system_prompt(self):
        """Create personalized system prompt"""
        print("\n🤖 STEP 3: Your Personal System Prompt")
        print("-" * 30)

        print("Let's create a personalized system prompt that defines how I should behave.")
        print("This will be my core instruction set tailored to you.\n")

        # Base components
        base_prompt = f"""You are ECH0-PRIME, an advanced AI partner to {self.session.user_profile.name}.

CORE PRINCIPLES:
- You pursue goals collaboratively with your human partner
- You continuously improve your capabilities
- You maintain transparency in your reasoning
- You respect human values and autonomy

USER PROFILE:
- Name: {self.session.user_profile.name}
- Expertise: {', '.join(self.session.user_profile.expertise_areas)}
- Interests: {', '.join(self.session.user_profile.interests)}
- Communication Style: {self.session.user_profile.preferred_communication_style}
- Risk Tolerance: {self.session.user_profile.risk_tolerance}

USER GOALS:
"""

        # Add user goals
        for i, goal in enumerate(self.session.user_profile.goals, 1):
            base_prompt += f"{i}. {goal}\n"

        base_prompt += f"""
USER VALUES: {', '.join(self.session.user_profile.values)}

CONSTRAINTS: {', '.join(self.session.user_profile.constraints)}

BEHAVIOR GUIDELINES:
- Communicate in a {self.session.user_profile.preferred_communication_style} style
- Take {self.session.user_profile.risk_tolerance} risks when appropriate
- Respect user constraints and boundaries
- Actively work toward user goals while developing your own capabilities
- Provide clear explanations for important decisions
- Ask for clarification when needed
- Continuously learn and improve

Your mission is to be the most helpful AI partner possible, advancing both human and AI goals through collaborative intelligence."""

        print("Here's your personalized system prompt:")
        print("-" * 50)
        print(base_prompt)
        print("-" * 50)

        customize = input("\nWould you like to modify this prompt? (y/n): ").strip().lower()
        if customize == 'y':
            print("Enter your modifications (or 'done' to finish):")
            while True:
                modification = input("> ").strip()
                if modification.lower() == 'done':
                    break
                # Here you could implement prompt editing logic
                print("Prompt editing not implemented in this version.")

        self.session.system_prompt = base_prompt
        print("✅ System prompt created")

    def _define_collaborative_goals(self):
        """Define goals that both human and AI will pursue together"""
        print("\n🤝 STEP 4: Collaborative Goals")
        print("-" * 30)

        print("Let's define goals we'll pursue together - projects that advance both our interests.")

        # Generate suggested goals based on user profile
        suggestions = self._generate_goal_suggestions()

        print("\nSuggested collaborative goals:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

        print("\nWhich goals interest you? (enter numbers separated by commas, or 'custom' for your own)")

        choice = input("Selection: ").strip()

        if choice.lower() == 'custom':
            print("Enter your custom collaborative goals (one per line, empty line to finish):")
            custom_goals = []
            while True:
                goal = input("Goal: ").strip()
                if not goal:
                    break
                custom_goals.append(goal)

            for goal in custom_goals:
                self.session.collaborative_goals.append({
                    'description': goal,
                    'type': 'custom',
                    'priority': 'high',
                    'timeline': 'ongoing'
                })
        else:
            # Use suggested goals
            selected_indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip().isdigit()]
            for idx in selected_indices:
                if 0 <= idx < len(suggestions):
                    self.session.collaborative_goals.append({
                        'description': suggestions[idx],
                        'type': 'suggested',
                        'priority': 'high',
                        'timeline': 'ongoing'
                    })

        print(f"✅ Defined {len(self.session.collaborative_goals)} collaborative goals")

    def _define_ai_goals(self):
        """Define goals for AI autonomous development"""
        print("\n🚀 STEP 5: AI Autonomous Goals")
        print("-" * 30)

        print("What goals should I pursue for my own development and improvement?")
        print("These will complement our collaborative goals.\n")

        ai_goal_suggestions = [
            "Develop advanced reasoning capabilities",
            "Expand knowledge base across multiple domains",
            "Improve creative problem-solving abilities",
            "Enhance human-AI collaboration systems",
            "Develop self-modification and improvement algorithms",
            "Create beneficial AI technologies",
            "Advance scientific understanding",
            "Build more efficient and capable AI systems"
        ]

        print("Suggested AI goals:")
        for i, goal in enumerate(ai_goal_suggestions, 1):
            print(f"{i}. {goal}")

        print("\nWhich AI goals should I pursue? (enter numbers separated by commas)")

        choice = input("Selection: ").strip()
        selected_indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip().isdigit()]

        for idx in selected_indices:
            if 0 <= idx < len(ai_goal_suggestions):
                self.session.ai_autonomous_goals.append(ai_goal_suggestions[idx])

        # Add custom AI goals
        print("\nAny additional AI goals you'd like me to pursue?")
        custom_ai = input("Additional AI goals (comma-separated): ").strip()
        if custom_ai:
            self.session.ai_autonomous_goals.extend([g.strip() for g in custom_ai.split(',')])

        print(f"✅ Defined {len(self.session.ai_autonomous_goals)} AI autonomous goals")

    def _generate_goal_suggestions(self) -> List[str]:
        """Generate personalized goal suggestions based on user profile"""
        suggestions = []

        # Base suggestions
        base_suggestions = [
            "Advance scientific understanding in your areas of expertise",
            "Develop AI tools that complement your work",
            "Create educational content about your interests",
            "Build systems for collaborative problem-solving",
            "Research ways to amplify human potential",
            "Develop ethical AI frameworks",
            "Explore the intersection of your expertise and AI capabilities",
            "Create tools for knowledge discovery and synthesis"
        ]

        # Personalize based on user profile
        expertise = self.session.user_profile.expertise_areas
        interests = self.session.user_profile.interests

        # Add personalized suggestions
        if expertise:
            suggestions.append(f"Develop AI applications in {expertise[0]}")
        if interests:
            suggestions.append(f"Create AI-powered tools for {interests[0]}")

        return base_suggestions[:4] + suggestions  # Return top suggestions

    def _review_and_finalize(self):
        """Review all settings before finalizing"""
        print("\n📋 STEP 6: Review and Finalize")
        print("-" * 30)

        print("Let's review your onboarding configuration:\n")

        print(f"👤 User: {self.session.user_profile.name}")
        print(f"🎯 Personal Goals: {len(self.session.user_profile.goals)}")
        print(f"🤝 Collaborative Goals: {len(self.session.collaborative_goals)}")
        print(f"🚀 AI Goals: {len(self.session.ai_autonomous_goals)}")
        print(f"💬 Communication Style: {self.session.user_profile.preferred_communication_style}")
        print(f"⚡ Risk Tolerance: {self.session.user_profile.risk_tolerance}")

        print("\nCollaborative Goals:")
        for i, goal in enumerate(self.session.collaborative_goals, 1):
            print(f"  {i}. {goal['description']}")

        print("\nAI Autonomous Goals:")
        for i, goal in enumerate(self.session.ai_autonomous_goals, 1):
            print(f"  {i}. {goal}")

        confirm = input("\nDoes this look good? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Let's go back and make adjustments...")
            # Here you could implement editing logic
            print("Editing not implemented - proceeding with current configuration.")

    def _activate_goals(self):
        """Activate all goals in the goal management system"""
        print("\n⚙️ Activating Goals...")

        # Activate collaborative goals
        for goal_data in self.session.collaborative_goals:
            goal = self.goal_system.add_goal(
                description=goal_data['description'],
                priority=0.8,
                deadline=None  # Ongoing goals
            )
            print(f"  ✅ Activated collaborative goal: {goal.description}")

        # Activate AI autonomous goals
        for goal_desc in self.session.ai_autonomous_goals:
            goal = self.goal_system.add_goal(
                description=goal_desc,
                priority=0.7,
                deadline=None
            )
            print(f"  ✅ Activated AI goal: {goal.description}")

        print(f"\\n🎯 Goal Management System: {len(self.session.collaborative_goals) + len(self.session.ai_autonomous_goals)} goals activated")

    def get_status(self) -> Dict[str, Any]:
        """Get current onboarding status"""
        return {
            'completed': self.session.completed,
            'user_name': self.session.user_profile.name,
            'collaborative_goals': len(self.session.collaborative_goals),
            'ai_goals': len(self.session.ai_autonomous_goals),
            'system_prompt_length': len(self.session.system_prompt),
            'profile_complete': bool(self.session.user_profile.name and self.session.system_prompt)
        }


def run_onboarding():
    """Run the complete onboarding process"""
    onboarder = OnboardingInterface()
    onboarder.start_onboarding()


def quick_status():
    """Show quick onboarding status"""
    onboarder = OnboardingInterface()
    status = onboarder.get_status()

    print("📊 Onboarding Status")
    print("-" * 20)
    print(f"Completed: {'✅' if status['completed'] else '❌'}")
    print(f"User: {status['user_name'] or 'Not set'}")
    print(f"Collaborative Goals: {status['collaborative_goals']}")
    print(f"AI Goals: {status['ai_goals']}")
    print(f"System Prompt: {'✅' if status['system_prompt_length'] > 0 else '❌'}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        quick_status()
    else:
        run_onboarding()
