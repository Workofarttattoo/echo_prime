#!/usr/bin/env python3
"""
Echo Lite Launcher

Quick start script for Echo Lite autonomous agent
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.agent_runtime import main

if __name__ == "__main__":
    print("🚀 Starting Echo Lite...")
    main()
