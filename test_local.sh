#!/bin/bash
# Test script for local federated learning setup
# This script helps test the setup on a single machine

echo "============================================================"
echo "Federated Learning Local Test"
echo "============================================================"
echo ""
echo "This will test the server and clients on your local machine."
echo "You'll need to run this in 3 separate terminals."
echo ""
echo "TERMINAL 1 - Start Server:"
echo "  python server.py"
echo ""
echo "TERMINAL 2 - Start Client 1 (after server starts):"
echo "  python client1.py localhost:8080"
echo ""
echo "TERMINAL 3 - Start Client 2 (after server starts):"
echo "  python client2.py localhost:8080"
echo ""
echo "============================================================"
echo ""
echo "Testing server startup (5 seconds)..."
echo ""

timeout 5 python server.py 2>&1 | head -30 || echo ""
echo ""
echo "If you saw server initialization messages above, the server is working!"
echo ""
echo "Now test the full setup by running the commands above in separate terminals."

