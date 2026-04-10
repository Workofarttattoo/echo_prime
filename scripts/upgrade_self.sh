#!/usr/bin/env bash
# ----------------------------------------------------------------------
# upgrade_self.sh – run Echo Prime’s self‑upgrade routine every 10 min
# and keep the logs directory tidy.
# ----------------------------------------------------------------------

# Activate the virtual‑env that Echo Prime uses
source /Users/noone/echo_prime/venv/bin/activate

# ----------------------------------------------------------------------
# 1️⃣ Run the upgrade routine
# ----------------------------------------------------------------------
# Replace the command below with the exact entry‑point you use for
# “self‑upgrade”.  Many Echo Prime setups use the gradual‑scaling
# orchestrator with an `--upgrade` flag – adjust as needed.
#
# Example:
#   python /Users/noone/echo_prime/gradual_scaling_orchestrator.py --upgrade
# ----------------------------------------------------------------------
/Users/noone/echo_prime/venv_scaling/bin/python /Users/noone/echo_prime/bbb_real/gradual_scaling_orchestrator.py
# Run autonomous improvement loops
python /Users/noone/echo_prime/missions/recursive_improvement.py || true
python /Users/noone/echo_prime/demo_continuous_self_improvement.py || true

# ----------------------------------------------------------------------
# 2️⃣ Prune old / oversized log files
# ----------------------------------------------------------------------
LOGDIR="/Users/noone/echo_prime/logs"

# • Delete log files older than 7 days (you can change the number)
find "$LOGDIR" -type f -mtime +7 -delete

# • Truncate any log that has grown beyond 100 MiB (keeps the file but
#   resets its size to zero).  Adjust the size limit if you prefer.
find "$LOGDIR" -type f -size +100M -exec truncate -s 0 {} \;

# ----------------------------------------------------------------------
# 3️⃣ Optional: keep a tiny “heartbeat” file so you can see the last run
# ----------------------------------------------------------------------
date +"%Y-%m-%d %H:%M:%S  →  upgrade & cleanup completed" >> "$LOGDIR/upgrade_self_heartbeat.log"
