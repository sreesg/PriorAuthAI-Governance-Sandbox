#!/bin/bash
# Kill any existing process on port 8000 and restart the server

echo "🛑 Stopping any existing server on port 8000..."
lsof -ti :8000 | xargs kill -9 2>/dev/null
sleep 1

echo "🚀 Starting PriorAuthAI server..."
cd "$(dirname "$0")"
python3 server.py
