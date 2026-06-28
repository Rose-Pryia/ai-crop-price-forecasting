"""
Test script to verify federated learning setup
This script tests the setup on a single machine
"""
import subprocess
import sys
import time
import signal
import os

def test_server_start():
    """Test if server can start"""
    print("=" * 60)
    print("Testing Server Startup")
    print("=" * 60)
    
    try:
        # Start server in background
        print("Starting server...")
        server_process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if process is still running
        if server_process.poll() is None:
            print("✅ Server started successfully!")
            print("⚠️  Server is running in background")
            print("   To stop it, you'll need to find the process and kill it")
            print("   Or press Ctrl+C if running in terminal")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print("❌ Server failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def test_imports():
    """Test if all required modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    modules = ['flwr', 'xgboost', 'pandas', 'numpy', 'sklearn', 'fastapi', 'uvicorn']
    all_ok = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            all_ok = False
    
    return all_ok

def test_data_files():
    """Test if data files exist"""
    print("\n" + "=" * 60)
    print("Testing Data Files")
    print("=" * 60)
    
    import os
    files = ['client1-data.csv', 'client2-data.csv', 'dataset.csv']
    all_ok = True
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    print("\n" + "=" * 60)
    print("Federated Learning Setup Test")
    print("=" * 60)
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data files
    data_ok = test_data_files()
    
    if not imports_ok or not data_ok:
        print("\n❌ Setup test failed. Please fix the issues above.")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ All checks passed!")
    print("=" * 60)
    print("\nTo test the full setup:")
    print("1. Open Terminal 1: python server.py")
    print("2. Open Terminal 2: python client1.py localhost:8080")
    print("3. Open Terminal 3: python client2.py localhost:8080")
    print("\nOr test with ngrok:")
    print("1. Open Terminal 1: python server.py --use-ngrok")
    print("2. Copy the ngrok URL shown")
    print("3. Open Terminal 2: python client1.py NGROK_URL")
    print("4. Open Terminal 3: python client2.py NGROK_URL")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

