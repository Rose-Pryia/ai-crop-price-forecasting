#!/bin/bash
# Script to run the federated learning server

echo "Starting Federated Learning Server..."
echo ""
echo "Options:"
echo "  Local IP:  ./run_server.sh"
echo "  Ngrok:     ./run_server.sh --use-ngrok"
echo ""

# Check if ngrok flag is provided
if [ "$1" == "--use-ngrok" ]; then
    echo "Starting server with ngrok tunnel..."
    python server.py --use-ngrok
else
    echo "Starting server with local IP..."
    echo "Server will listen on 0.0.0.0:8080"
    echo "Clients can connect using your local IP address"
    echo ""
    python server.py
fi

