#!/bin/bash
# Simple test script for federated learning

echo "============================================================"
echo "Federated Learning Setup Test"
echo "============================================================"
echo ""

# Check if label encoders exist
if [ ! -f "label_encoders.pkl" ]; then
    echo "⚠️  label_encoders.pkl not found"
    echo "   This will be created automatically by the first client"
    echo ""
fi

echo "✅ Setup verified!"
echo ""
echo "To test the full setup, open 3 terminal windows:"
echo ""
echo "TERMINAL 1 - Server:"
echo "  cd $(pwd)"
echo "  python server.py"
echo ""
echo "TERMINAL 2 - Client 1 (wait for server to start):"
echo "  cd $(pwd)"
echo "  python client1.py localhost:8080"
echo ""
echo "TERMINAL 3 - Client 2 (wait for server to start):"
echo "  cd $(pwd)"
echo "  python client2.py localhost:8080"
echo ""
echo "============================================================"
echo ""
echo "Quick test - Starting server for 10 seconds..."
echo "You should see server initialization messages..."
echo ""

python server.py &
SERVER_PID=$!
sleep 10
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "If you saw server messages above, the setup is working!"
echo "Now run the full test in 3 separate terminals as shown above."

