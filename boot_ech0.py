#!/usr/bin/env python3
"""
ECH0-PRIME Boot Script
Clean startup for the ECH0-PRIME AGI system
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main boot function"""
    print("🧠 ECH0-PRIME Cognitive-Synthetic Architecture")
    print("🍎 Enhanced with Apple Intelligence Integration")
    print("=" * 60)

    try:
        # Import and boot the system
        from main_orchestrator import boot_system
        boot_system()

    except KeyboardInterrupt:
        print("\n\n👋 ECH0-PRIME shutdown requested by user")
        print("System powered down successfully")
    except Exception as e:
        print(f"\n❌ Boot error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
