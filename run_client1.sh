#!/bin/bash
# Script to run Client 1 federated learning client

# Get server IP from command line or use localhost
SERVER_IP=${1:-localhost}

echo "Starting Client 1..."
echo "Connecting to server at ${SERVER_IP}:8080"
echo ""

# If server IP is not localhost, use it
if [ "$SERVER_IP" != "localhost" ]; then
    python client1.py ${SERVER_IP}:8080
else
    python client1.py
fi

