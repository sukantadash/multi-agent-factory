#!/bin/bash
# Intelligent Operations Agent (Phase-1) - Execution Script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Optional: Apply test pod (uncomment if needed)
# oc apply -f oom-test-pod.yaml

# Run the LangGraph-based intelligent agent
# Usage: ./script.sh [namespace]
# Example: ./script.sh oom-test
python intelligent_agent_phase1.py "$@"