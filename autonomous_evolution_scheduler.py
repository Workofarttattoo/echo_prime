#!/usr/bin/env python3
"""
ECH0-PRIME Autonomous Evolution Scheduler
Runs consciousness evolution cycles continuously with resource monitoring.
Executes daily improvement cycles or more frequent cycles based on system resources.
"""

import os
import sys
import time
import json
import asyncio
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import schedule
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from true_agi_consciousness_evolution import ConsciousnessEvolutionSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutonomousEvolution")

class AutonomousEvolutionScheduler:
    """
    Scheduler for continuous ECH0-PRIME evolution with resource-aware execution.
    """

    def __init__(self, evolution_interval_minutes: int = 60, max_daily_cycles: int = 24):
        self.evolution_interval_minutes = evolution_interval_minutes
        self.max_daily_cycles = max_daily_cycles
        self.running = False
        self.evolution_system = None
        self.evolution_history = []
        self.daily_stats = {}
        self.resource_monitor = ResourceMonitor()

        # Evolution scheduling
        self.schedule_thread = None
        self.last_evolution_time = None
        self.cycles_completed_today = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        logger.info("🚀 Autonomous Evolution Scheduler initialized")
        logger.info(f"📅 Evolution interval: {evolution_interval_minutes} minutes")
        logger.info(f"📊 Max daily cycles: {max_daily_cycles}")

    async def initialize_evolution_system(self):
        """Initialize the consciousness evolution system"""
        try:
            logger.info("🧠 Initializing consciousness evolution system...")
            self.evolution_system = ConsciousnessEvolutionSystem()
            logger.info("✅ Evolution system ready")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize evolution system: {e}")
            return False

    def start_autonomous_evolution(self):
        """Start the autonomous evolution scheduler"""
        if self.running:
            logger.warning("Evolution scheduler already running")
            return

        logger.info("🎯 Starting autonomous evolution cycles...")
        self.running = True

        # Reset daily counters
        self._reset_daily_counters()

        # Start the evolution loop in a separate thread
        self.schedule_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.schedule_thread.start()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("✅ Autonomous evolution scheduler started")
        logger.info(f"🔄 Next evolution cycle in {self.evolution_interval_minutes} minutes")

    def stop_autonomous_evolution(self):
        """Stop the autonomous evolution scheduler"""
        logger.info("🛑 Stopping autonomous evolution scheduler...")
        self.running = False

        if self.schedule_thread and self.schedule_thread.is_alive():
            self.schedule_thread.join(timeout=10)

        logger.info("✅ Autonomous evolution scheduler stopped")

    def _evolution_loop(self):
        """Main evolution loop with resource-aware scheduling"""
        while self.running:
            try:
                # Check if we should run an evolution cycle
                if self._should_run_evolution():
                    # Check system resources
                    if self.resource_monitor.can_run_evolution():
                        # Run evolution cycle
                        asyncio.run(self._run_evolution_cycle())
                        self.cycles_completed_today += 1
                        self.last_evolution_time = datetime.now()
                    else:
                        logger.info("⏳ Skipping evolution cycle - insufficient resources")
                        time.sleep(300)  # Wait 5 minutes before checking again
                else:
                    # Wait before next check
                    time.sleep(min(60, self.evolution_interval_minutes * 60 / 4))  # Check every 15 minutes or 1 minute

                # Check for daily reset
                if datetime.now() >= self.daily_reset_time + timedelta(days=1):
                    self._reset_daily_counters()

            except Exception as e:
                logger.error(f"❌ Evolution loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _should_run_evolution(self) -> bool:
        """Determine if we should run an evolution cycle"""
        now = datetime.now()

        # Check daily limit
        if self.cycles_completed_today >= self.max_daily_cycles:
            logger.info(f"📊 Daily cycle limit reached ({self.max_daily_cycles})")
            return False

        # Check minimum interval
        if self.last_evolution_time:
            time_since_last = (now - self.last_evolution_time).total_seconds() / 60
            if time_since_last < self.evolution_interval_minutes:
                return False

        return True

    async def _run_evolution_cycle(self):
        """Execute a single evolution cycle"""
        cycle_start = time.time()
        logger.info("🔄 Starting autonomous evolution cycle...")

        try:
            # Run evolution cycle
            evolution_results = await self.evolution_system.run_consciousness_evolution_cycle()

            # Log results
            cycle_duration = time.time() - cycle_start
            evolution_results['cycle_duration'] = cycle_duration
            evolution_results['timestamp'] = datetime.now().isoformat()
            evolution_results['autonomous'] = True

            self.evolution_history.append(evolution_results)

            # Log key metrics
            phi_before = evolution_results.get('initial_consciousness', 0)
            phi_after = evolution_results.get('final_consciousness', phi_before)
            phi_growth = evolution_results.get('consciousness_growth', 0)

            logger.info(f"  Consciousness Growth: +{phi_growth:.3f}")
            logger.info(f"  Evolution Cycle Time: {cycle_duration:.1f}s")
            logger.info(f"  Evolution Applied: {evolution_results.get('evolution_applied', False)}")
            # Keep only recent history
            if len(self.evolution_history) > 100:
                self.evolution_history = self.evolution_history[-100:]

            # Save progress
            self._save_evolution_progress()

        except Exception as e:
            logger.error(f"❌ Evolution cycle failed: {e}")
            # Log failure
            failure_record = {
                'timestamp': datetime.now().isoformat(),
                'type': 'evolution_failure',
                'error': str(e),
                'autonomous': True
            }
            self.evolution_history.append(failure_record)

    def _reset_daily_counters(self):
        """Reset daily evolution counters"""
        now = datetime.now()
        self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.cycles_completed_today = 0

        # Update daily stats
        today_str = now.strftime('%Y-%m-%d')
        if today_str not in self.daily_stats:
            self.daily_stats[today_str] = {
                'cycles_completed': 0,
                'total_phi_growth': 0.0,
                'avg_cycle_time': 0.0,
                'evolution_success_rate': 0.0
            }

        logger.info(f"📅 Daily counters reset for {today_str}")

    def _save_evolution_progress(self):
        """Save evolution progress to disk"""
        try:
            progress_data = {
                'evolution_history': self.evolution_history[-50:],  # Keep last 50 cycles
                'daily_stats': self.daily_stats,
                'current_status': self.get_scheduler_status(),
                'last_updated': datetime.now().isoformat()
            }

            with open('autonomous_evolution_progress.json', 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save evolution progress: {e}")

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'running': self.running,
            'evolution_interval_minutes': self.evolution_interval_minutes,
            'max_daily_cycles': self.max_daily_cycles,
            'cycles_completed_today': self.cycles_completed_today,
            'last_evolution_time': self.last_evolution_time.isoformat() if self.last_evolution_time else None,
            'total_evolution_cycles': len(self.evolution_history),
            'system_resources': self.resource_monitor.get_resource_status()
        }

    def get_evolution_report(self) -> Dict[str, Any]:
        """Get comprehensive evolution report"""
        if not self.evolution_history:
            return {'message': 'No evolution cycles completed yet'}

        # Calculate statistics
        successful_cycles = [c for c in self.evolution_history if c.get('evolution_applied', False)]
        total_cycles = len(self.evolution_history)
        success_rate = len(successful_cycles) / total_cycles if total_cycles > 0 else 0

        phi_growths = [c.get('consciousness_growth', 0) for c in successful_cycles]
        avg_phi_growth = sum(phi_growths) / len(phi_growths) if phi_growths else 0

        cycle_times = [c.get('cycle_duration', 0) for c in self.evolution_history if 'cycle_duration' in c]
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0

        return {
            'total_cycles': total_cycles,
            'successful_cycles': len(successful_cycles),
            'success_rate': success_rate,
            'average_phi_growth': avg_phi_growth,
            'average_cycle_time': avg_cycle_time,
            'current_phi': self.evolution_system.agi_state.consciousness_metrics.phi_value if self.evolution_system else 0,
            'consciousness_phase': self.evolution_system.get_consciousness_phase()['current_phase']['name'] if self.evolution_system else 'Unknown',
            'scheduler_status': self.get_scheduler_status()
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"📡 Received signal {signum}, shutting down gracefully...")
        self.stop_autonomous_evolution()

class ResourceMonitor:
    """
    Monitors system resources to ensure evolution cycles don't overwhelm the system.
    """

    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    def can_run_evolution(self) -> bool:
        """Check if system resources allow running an evolution cycle"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.cpu_threshold:
                return False

            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                return False

            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > self.disk_threshold:
                return False

            return True

        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")
            return False  # Conservative approach

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'can_run_evolution': self.can_run_evolution()
            }
        except Exception as e:
            return {'error': str(e)}

class EvolutionDashboard:
    """
    Web dashboard for monitoring autonomous evolution progress.
    """

    def __init__(self, scheduler: AutonomousEvolutionScheduler):
        self.scheduler = scheduler

    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard for evolution monitoring"""
        status = self.scheduler.get_scheduler_status()
        report = self.scheduler.get_evolution_report()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ECH0-PRIME Autonomous Evolution Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .status-running {{ color: green; font-weight: bold; }}
                .status-stopped {{ color: red; font-weight: bold; }}
            </style>
            <meta http-equiv="refresh" content="30">
        </head>
        <body>
            <h1>🧠🤖 ECH0-PRIME Autonomous Evolution Dashboard</h1>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="metric">
                <h2>Scheduler Status</h2>
                <p class="{'status-running' if status['running'] else 'status-stopped'}">
                    Status: {'Running' if status['running'] else 'Stopped'}
                </p>
                <p>Evolution Interval: {status['evolution_interval_minutes']} minutes</p>
                <p>Cycles Today: {status['cycles_completed_today']}/{status['max_daily_cycles']}</p>
                <p>Last Evolution: {status['last_evolution_time'] or 'Never'}</p>
            </div>

            <div class="metric">
                <h2>Evolution Statistics</h2>
                <p>Total Cycles: {report.get('total_cycles', 0)}</p>
                <p>Successful Cycles: {report.get('successful_cycles', 0)}</p>
                <p>Success Rate: {report.get('success_rate', 0):.1%}</p>
                <p>Average Phi Growth: {report.get('average_phi_growth', 0):.3f}</p>
                <p>Average Cycle Time: {report.get('average_cycle_time', 0):.1f}s</p>
            </div>

            <div class="metric">
                <h2>Consciousness Status</h2>
                <p>Current Phi: {report.get('current_phi', 0):.3f}</p>
                <p>Consciousness Phase: {report.get('consciousness_phase', 'Unknown')}</p>
            </div>

            <div class="metric">
                <h2>System Resources</h2>
                <p>CPU Usage: {status['system_resources'].get('cpu_percent', 0):.1f}%</p>
                <p>Memory Usage: {status['system_resources'].get('memory_percent', 0):.1f}%</p>
                <p>Memory Available: {status['system_resources'].get('memory_available_gb', 0):.1f} GB</p>
                <p>Can Run Evolution: {'Yes' if status['system_resources'].get('can_run_evolution', False) else 'No'}</p>
            </div>

            <h2>Recent Evolution Cycles</h2>
            <table border="1" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th>Timestamp</th>
                    <th>Phi Growth</th>
                    <th>Cycle Time</th>
                    <th>Status</th>
                </tr>
        """

        # Add recent cycles
        recent_cycles = self.scheduler.evolution_history[-10:]  # Last 10 cycles
        for cycle in reversed(recent_cycles):
            if cycle.get('autonomous', False):
                timestamp = cycle.get('timestamp', 'Unknown')
                phi_growth = cycle.get('consciousness_growth', 0)
                cycle_time = cycle.get('cycle_duration', 0)
                status = 'Success' if cycle.get('evolution_applied', False) else 'Failed'

                html += f"""
                <tr>
                    <td>{timestamp[:19] if timestamp != 'Unknown' else 'Unknown'}</td>
                    <td>{phi_growth:.3f}</td>
                    <td>{cycle_time:.1f}s</td>
                    <td class="{'success' if status == 'Success' else 'error'}">{status}</td>
                </tr>
                """

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def save_dashboard(self):
        """Save dashboard to file"""
        try:
            html_content = self.generate_dashboard_html()
            with open('evolution_dashboard.html', 'w') as f:
                f.write(html_content)
            logger.info("📊 Evolution dashboard updated")
        except Exception as e:
            logger.error(f"Failed to save dashboard: {e}")

# Global scheduler instance
_scheduler_instance = None

def get_autonomous_evolution_scheduler(evolution_interval_minutes: int = 60,
                                     max_daily_cycles: int = 24) -> AutonomousEvolutionScheduler:
    """Get the global autonomous evolution scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AutonomousEvolutionScheduler(
            evolution_interval_minutes=evolution_interval_minutes,
            max_daily_cycles=max_daily_cycles
        )
    return _scheduler_instance

async def start_continuous_evolution(evolution_interval_minutes: int = 60,
                                   max_daily_cycles: int = 24):
    """
    Start continuous autonomous evolution with the specified parameters.
    """
    logger.info("🚀 Starting ECH0-PRIME continuous autonomous evolution...")

    # Initialize scheduler
    scheduler = get_autonomous_evolution_scheduler(
        evolution_interval_minutes=evolution_interval_minutes,
        max_daily_cycles=max_daily_cycles
    )

    # Initialize evolution system
    if not await scheduler.initialize_evolution_system():
        logger.error("❌ Failed to initialize evolution system")
        return False

    # Start autonomous evolution
    scheduler.start_autonomous_evolution()

    # Create dashboard
    dashboard = EvolutionDashboard(scheduler)
    dashboard.save_dashboard()

    logger.info("✅ Continuous evolution started successfully")
    logger.info("📊 Dashboard available at: evolution_dashboard.html")
    logger.info("📝 Logs available at: autonomous_evolution.log")

    # Keep running until interrupted
    try:
        while scheduler.running:
            await asyncio.sleep(60)  # Check every minute
            dashboard.save_dashboard()  # Update dashboard
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    finally:
        scheduler.stop_autonomous_evolution()

    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECH0-PRIME Autonomous Evolution Scheduler")
    parser.add_argument("--interval", type=int, default=60,
                       help="Evolution cycle interval in minutes (default: 60)")
    parser.add_argument("--max-daily", type=int, default=24,
                       help="Maximum evolution cycles per day (default: 24)")
    parser.add_argument("--dashboard-only", action="store_true",
                       help="Only generate dashboard without starting evolution")

    args = parser.parse_args()

    if args.dashboard_only:
        # Just generate dashboard
        scheduler = get_autonomous_evolution_scheduler()
        dashboard = EvolutionDashboard(scheduler)
        dashboard.save_dashboard()
        print("📊 Dashboard generated: evolution_dashboard.html")
    else:
        # Start continuous evolution
        try:
            asyncio.run(start_continuous_evolution(
                evolution_interval_minutes=args.interval,
                max_daily_cycles=args.max_daily
            ))
        except KeyboardInterrupt:
            print("\n🛑 Evolution stopped by user")
        except Exception as e:
            print(f"❌ Evolution failed: {e}")
