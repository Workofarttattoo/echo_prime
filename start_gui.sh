#!/bin/bash
# ECH0-PRIME GUI Startup Script
# Starts the dashboard GUI on port 8080

echo "🚀 Starting ECH0-PRIME GUI Dashboard..."
echo "========================================="

# Navigate to dashboard directory
cd dashboard/v2/dist

# Start the HTTP server
echo "🌐 Serving dashboard on http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

python3 -m http.server 8080
