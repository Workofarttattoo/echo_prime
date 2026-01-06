#!/bin/bash
# Backup evolution data and logs
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp autonomous_evolution.log "$BACKUP_DIR/" 2>/dev/null || true
cp autonomous_evolution_progress.json "$BACKUP_DIR/" 2>/dev/null || true
cp evolution_dashboard.html "$BACKUP_DIR/" 2>/dev/null || true

echo "Evolution data backed up to: $BACKUP_DIR"
