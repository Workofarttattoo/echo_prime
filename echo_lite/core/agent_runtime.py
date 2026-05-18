"""
Autonomous Agent Runtime for Echo Lite

Runs continuously like Hermes agent with:
- Persistent identity and memory
- Autonomous decision-making
- Task execution
- Proactive behavior
- Real-time responsiveness
"""

import time
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .echo_lite import EchoLite, EchoLiteConfig
from .persistent_memory import PersistentMemory


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    LEARNING = "learning"
    SLEEPING = "sleeping"


@dataclass
class Task:
    """Task for agent to execute"""
    task_id: str
    description: str
    priority: int = 5
    deadline: Optional[float] = None
    status: str = "pending"
    result: Optional[Any] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class AutonomousAgent:
    """
    Autonomous agent runtime with continuous operation

    Features:
    - Runs continuously in background
    - Persistent memory across reboots
    - Proactive task execution
    - Real-time responsiveness
    - Identity continuity
    """

    def __init__(
        self,
        config: Optional[EchoLiteConfig] = None,
        memory_db: str = "echo_lite_agent.db"
    ):
        print("\n" + "="*70)
        print("🤖 ECHO LITE AUTONOMOUS AGENT")
        print("="*70)

        # Initialize Echo Lite cognition
        self.echo = EchoLite(config)

        # Initialize persistent memory
        self.memory = PersistentMemory(memory_db)

        # Agent state
        self.state = AgentState.IDLE
        self.running = True

        # Task queue
        self.task_queue = queue.PriorityQueue()
        self.current_task = None

        # Background threads
        self.cognitive_thread = None
        self.task_thread = None

        # Performance tracking
        self.cycles_per_second = 0
        self.last_cycle_time = time.time()

        # Load previous state
        self._restore_state()

        print(f"Identity: {self.memory.identity['name']}")
        print(f"Age: {self._get_age_string()}")
        print(f"Total memories: {self.memory.count_memories()}")
        print(f"Total cycles: {self.memory.identity.get('total_cycles', 0)}")
        print("="*70 + "\n")

    def _restore_state(self):
        """Restore state from previous session"""
        # Load cognitive state
        saved_state = self.memory.load_latest_cognitive_state()

        if saved_state:
            print(f"♻️  Restored state from {self._format_time_ago(saved_state['timestamp'])}")

            # Restore cycle count
            self.echo.cycle_count = saved_state['cycle_count']

            # Store restoration memory
            self.memory.store_memory(
                f"Restored from previous session (cycle {saved_state['cycle_count']})",
                memory_type="identity",
                importance=0.7
            )
        else:
            print("🆕 Starting fresh session")

            # Store birth memory
            self.memory.store_memory(
                "Agent initialized for first time",
                memory_type="identity",
                importance=1.0,
                metadata={'event': 'birth'}
            )

    def _get_age_string(self) -> str:
        """Get human-readable age"""
        birth_time = self.memory.identity.get('birth_timestamp', time.time())
        age_seconds = time.time() - birth_time

        if age_seconds < 3600:
            return f"{age_seconds/60:.0f} minutes"
        elif age_seconds < 86400:
            return f"{age_seconds/3600:.1f} hours"
        else:
            return f"{age_seconds/86400:.1f} days"

    def _format_time_ago(self, timestamp: float) -> str:
        """Format time ago"""
        seconds = time.time() - timestamp
        if seconds < 60:
            return f"{seconds:.0f}s ago"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m ago"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h ago"
        else:
            return f"{seconds/86400:.1f}d ago"

    def start(self):
        """Start autonomous operation"""
        print("🚀 Starting autonomous agent...\n")

        # Start cognitive loop
        self.cognitive_thread = threading.Thread(
            target=self._cognitive_loop,
            daemon=True
        )
        self.cognitive_thread.start()

        # Start task executor
        self.task_thread = threading.Thread(
            target=self._task_loop,
            daemon=True
        )
        self.task_thread.start()

        print("✅ Agent running autonomously")
        print("   Press Ctrl+C to stop\n")

    def _cognitive_loop(self):
        """
        Continuous cognitive processing loop

        Runs in background, processing inputs and updating state
        """
        while self.running:
            try:
                cycle_start = time.time()

                # Update state
                self.state = AgentState.THINKING

                # Recall recent context from memory
                recent_memories = self.memory.recall_memories(
                    memory_type="episodic",
                    limit=5
                )

                context = " ".join([m['content'] for m in recent_memories[-3:]])

                if not context:
                    context = "Idle, awaiting input"

                # Cognitive cycle
                result = self.echo.process(context)

                # Store cognitive state periodically
                if self.echo.cycle_count % 100 == 0:
                    self.memory.save_cognitive_state(
                        state_vector=result['cognitive']['sensory'],
                        cycle_count=self.echo.cycle_count,
                        metadata={'state': self.state.value}
                    )

                # Update identity
                self.memory.identity['total_cycles'] = self.echo.cycle_count
                if self.echo.cycle_count % 1000 == 0:
                    self.memory.update_identity(self.memory.identity)

                # Calculate performance
                cycle_time = time.time() - cycle_start
                self.cycles_per_second = 1.0 / cycle_time if cycle_time > 0 else 0

                # Idle state
                self.state = AgentState.IDLE

                # Brief sleep to avoid CPU overload
                time.sleep(0.1)

            except Exception as e:
                print(f"⚠️  Cognitive loop error: {e}")
                time.sleep(1)

    def _task_loop(self):
        """
        Task execution loop

        Processes tasks from queue
        """
        while self.running:
            try:
                # Get task (blocking with timeout)
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Execute task
                self.state = AgentState.EXECUTING
                self.current_task = task

                print(f"\n🎯 Executing: {task.description}")

                # Process task
                result = self.echo.process(task.description)

                # Store task memory
                self.memory.store_memory(
                    f"Completed task: {task.description}",
                    memory_type="episodic",
                    importance=task.priority / 10.0,
                    metadata={
                        'task_id': task.task_id,
                        'priority': task.priority
                    }
                )

                task.status = "completed"
                task.result = result

                print(f"✅ Task completed\n")

                self.current_task = None
                self.state = AgentState.IDLE

            except Exception as e:
                print(f"⚠️  Task execution error: {e}")
                if self.current_task:
                    self.current_task.status = "failed"

    def submit_task(self, description: str, priority: int = 5) -> Task:
        """
        Submit task to agent

        Args:
            description: Task description
            priority: Priority (1-10, higher = more urgent)

        Returns:
            Task object
        """
        task = Task(
            task_id=f"task_{int(time.time())}",
            description=description,
            priority=priority
        )

        # Higher priority = lower queue number (inverted)
        self.task_queue.put((10 - priority, task))

        print(f"📝 Task queued: {description} (priority {priority})")

        return task

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'state': self.state.value,
            'age': self._get_age_string(),
            'total_cycles': self.echo.cycle_count,
            'cycles_per_second': self.cycles_per_second,
            'memory_count': self.memory.count_memories(),
            'task_queue_size': self.task_queue.qsize(),
            'current_task': self.current_task.description if self.current_task else None,
            'identity': self.memory.identity['name'],
            'cognitive_state': self.echo.cognitive.get_state()
        }

    def interact(self, message: str) -> str:
        """
        Interact with agent

        Args:
            message: Input message

        Returns:
            Agent response
        """
        print(f"\n💬 User: {message}")

        # Store interaction as memory
        self.memory.store_memory(
            f"User said: {message}",
            memory_type="episodic",
            importance=0.7,
            metadata={'type': 'interaction'}
        )

        # Process
        result = self.echo.process(message)

        # Generate response based on cognitive state
        response = self._generate_response(message, result)

        # Store response
        self.memory.store_memory(
            f"I responded: {response}",
            memory_type="episodic",
            importance=0.6
        )

        print(f"🤖 Agent: {response}\n")

        return response

    def _generate_response(self, input_msg: str, result: Dict[str, Any]) -> str:
        """Generate contextual response"""

        attention = result['cognitive']['reasoning']['attention']
        confidence = result['cognitive']['reasoning']['confidence']

        # Simple response generation (can be enhanced with LLM)
        if "status" in input_msg.lower():
            status = self.get_status()
            return f"I am {status['state']}, {status['age']} old. " \
                   f"I have {status['memory_count']} memories and have completed {status['total_cycles']} cognitive cycles."

        elif "remember" in input_msg.lower() or "recall" in input_msg.lower():
            memories = self.memory.recall_memories(limit=3)
            if memories:
                return f"I remember: {memories[0]['content']}"
            else:
                return "I don't have any memories to recall yet."

        elif "who" in input_msg.lower() and "you" in input_msg.lower():
            return f"I am {self.memory.identity['name']}, an autonomous cognitive agent. " \
                   f"I have been operational for {self._get_age_string()}."

        else:
            # Default contextual response
            if confidence > 0.5:
                return f"Processing your request with {confidence:.1%} confidence. " \
                       f"My attention is at {attention:.1%}."
            else:
                return "I'm analyzing your input. Please give me a moment to process."

    def shutdown(self):
        """Graceful shutdown"""
        print("\n🛑 Shutting down agent...")

        self.running = False

        # Save final state
        if self.echo.cognitive.state:
            self.memory.save_cognitive_state(
                state_vector=self.echo.cognitive.state['level_2'],
                cycle_count=self.echo.cycle_count,
                metadata={'event': 'shutdown'}
            )

        # Store shutdown memory
        self.memory.store_memory(
            f"Agent shutdown after {self.echo.cycle_count} cycles",
            memory_type="identity",
            importance=0.8,
            metadata={'event': 'shutdown'}
        )

        # Update identity
        self.memory.update_identity(self.memory.identity)

        # Wait for threads
        if self.cognitive_thread:
            self.cognitive_thread.join(timeout=2)
        if self.task_thread:
            self.task_thread.join(timeout=2)

        print("✅ Agent shut down gracefully")
        print(f"   Total cycles: {self.echo.cycle_count}")
        print(f"   Total memories: {self.memory.count_memories()}\n")


def main():
    """Main entry point for autonomous agent"""

    # Create agent
    agent = AutonomousAgent()

    # Start autonomous operation
    agent.start()

    # Interactive loop
    print("🎮 Interactive Mode")
    print("Commands: status, task <description>, quit\n")

    try:
        while True:
            user_input = input("→ ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            elif user_input.lower() == "status":
                status = agent.get_status()
                print("\n📊 Agent Status:")
                for key, value in status.items():
                    if key != 'cognitive_state':
                        print(f"   {key}: {value}")
                print()

            elif user_input.lower().startswith("task "):
                description = user_input[5:]
                agent.submit_task(description, priority=7)

            else:
                agent.interact(user_input)

    except KeyboardInterrupt:
        print("\n")

    # Shutdown
    agent.shutdown()


if __name__ == "__main__":
    main()
