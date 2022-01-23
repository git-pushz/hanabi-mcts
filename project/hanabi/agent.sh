#!/bin/bash

clear

agent_name="AI$((1 + RANDOM % 10))"
if [ $# -lt 1 ]; then
    echo "Missing agent name"
    echo "Using name $agent_name"
    elif [ $# -gt 1 ]; then
        echo "Too many arguments"
        exit 255
fi
agent_name=$1

python3 agent-client.py 127.0.0.1 1024 "$agent_name"