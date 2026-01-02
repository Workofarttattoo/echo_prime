#!/usr/bin/env python3
"""
ECH0-PRIME Onboarding Runner

Simple interface to run the onboarding process and manage user profiles.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main onboarding interface"""
    print("🧠 ECH0-PRIME Onboarding System")
    print("=" * 40)

    while True:
        print("\nAvailable options:")
        print("1. Start new onboarding")
        print("2. View current profile")
        print("3. Update profile")
        print("4. View active goals")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == '1':
            run_full_onboarding()
        elif choice == '2':
            show_profile()
        elif choice == '3':
            update_profile()
        elif choice == '4':
            show_goals()
        elif choice == '5':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please select 1-5.")

def run_full_onboarding():
    """Run the complete onboarding process"""
    print("\n🚀 Starting ECH0-PRIME Onboarding...")
    print("This will take 10-15 minutes to complete properly.")
    confirm = input("Continue? (y/n): ").strip().lower()

    if confirm == 'y':
        from onboarding import run_onboarding
        run_onboarding()
    else:
        print("Onboarding cancelled.")

def show_profile():
    """Show current user profile"""
    profile_file = "user_profile.json"

    if os.path.exists(profile_file):
        try:
            with open(profile_file, 'r') as f:
                profile = json.load(f)

            print("\n👤 Current User Profile")
            print("-" * 25)

            user_profile = profile.get('user_profile', {})
            print(f"Name: {user_profile.get('name', 'Not set')}")
            print(f"Goals: {len(user_profile.get('goals', []))}")
            print(f"Values: {', '.join(user_profile.get('values', []))}")
            print(f"Expertise: {', '.join(user_profile.get('expertise_areas', []))}")
            print(f"Communication: {user_profile.get('preferred_communication_style', 'Not set')}")
            print(f"Risk Tolerance: {user_profile.get('risk_tolerance', 'Not set')}")

            print(f"\n🤝 Collaborative Goals: {len(profile.get('collaborative_goals', []))}")
            for i, goal in enumerate(profile.get('collaborative_goals', []), 1):
                print(f"  {i}. {goal['description']}")

            print(f"\n🚀 AI Goals: {len(profile.get('ai_autonomous_goals', []))}")
            for i, goal in enumerate(profile.get('ai_autonomous_goals', []), 1):
                print(f"  {i}. {goal}")

            completed = profile.get('completed', False)
            print(f"\nStatus: {'✅ Complete' if completed else '❌ Incomplete'}")

        except Exception as e:
            print(f"Error reading profile: {e}")
    else:
        print("❌ No user profile found. Run onboarding first.")

def update_profile():
    """Update existing profile"""
    print("\n⚙️ Profile Update Options")
    print("-" * 25)
    print("1. Update personal information")
    print("2. Add new goals")
    print("3. Modify system prompt")
    print("4. Add collaborative goals")
    print("5. Back to main menu")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == '1':
        update_personal_info()
    elif choice == '2':
        add_goals()
    elif choice == '3':
        modify_system_prompt()
    elif choice == '4':
        add_collaborative_goals()
    elif choice == '5':
        return
    else:
        print("Invalid choice.")

def update_personal_info():
    """Update personal information"""
    print("Personal info update not implemented yet.")
    print("Please run full onboarding to make changes.")

def add_goals():
    """Add new goals"""
    print("Add goals not implemented yet.")
    print("Please run full onboarding to make changes.")

def modify_system_prompt():
    """Modify system prompt"""
    print("System prompt modification not implemented yet.")
    print("Please run full onboarding to make changes.")

def add_collaborative_goals():
    """Add collaborative goals"""
    print("Add collaborative goals not implemented yet.")
    print("Please run full onboarding to make changes.")

def show_goals():
    """Show active goals from the goal management system"""
    try:
        from missions.long_term_goals import LongTermGoalSystem
        goal_system = LongTermGoalSystem()
        goals = goal_system.get_all_goals()

        if goals:
            print("\n🎯 Active Goals")
            print("-" * 15)
            for i, goal in enumerate(goals, 1):
                status = "✅" if goal.completed else "🔄"
                print(f"{i}. {status} {goal.description}")
                print(f"   Priority: {goal.priority}, Progress: {getattr(goal, 'progress', 0):.1%}")
        else:
            print("❌ No active goals found.")
            print("Run onboarding to set up goals.")

    except Exception as e:
        print(f"Error accessing goals: {e}")

if __name__ == "__main__":
    main()
