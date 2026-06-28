"""
Full test of federated learning setup on a single machine
This will start the server and both clients to test the complete flow
"""
import subprocess
import sys
import time
import signal
import os
import threading

def start_server():
    """Start the server"""
    print("\n" + "="*60)
    print("Starting Server...")
    print("="*60)
    process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    return process

def start_client(client_id, server_address="localhost:8080", delay=5):
    """Start a client"""
    time.sleep(delay)  # Wait for server to be ready
    print(f"\n{'='*60}")
    print(f"Starting Client {client_id}...")
    print("="*60)
    process = subprocess.Popen(
        [sys.executable, f"client{client_id}.py", server_address],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    return process

def monitor_process(name, process, duration=30):
    """Monitor a process for a duration"""
    start_time = time.time()
    output_lines = []
    
    while time.time() - start_time < duration:
        if process.poll() is not None:
            # Process ended
            stdout, _ = process.communicate()
            output_lines.extend(stdout.split('\n') if stdout else [])
            print(f"\n{name} ended (exit code: {process.returncode})")
            break
        
        # Read output if available
        try:
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                # Print important lines
                if any(keyword in line.lower() for keyword in ['error', 'connected', 'round', 'training', 'aggregation']):
                    print(f"[{name}] {line.strip()}")
        except:
            pass
        
        time.sleep(0.1)
    
    return output_lines

def main():
    print("="*60)
    print("Federated Learning Full Setup Test")
    print("="*60)
    print("\nThis will:")
    print("1. Start the server")
    print("2. Start Client 1 (after 5 seconds)")
    print("3. Start Client 2 (after 8 seconds)")
    print("4. Monitor for 30 seconds")
    print("\nPress Ctrl+C to stop all processes\n")
    
    server_process = None
    client1_process = None
    client2_process = None
    
    try:
        # Start server
        server_process = start_server()
        time.sleep(3)  # Give server time to start
        
        # Check if server is running
        if server_process.poll() is not None:
            stdout, _ = server_process.communicate()
            print(f"\n❌ Server failed to start!")
            print(stdout)
            return 1
        
        print("✅ Server started")
        
        # Start clients
        client1_process = start_client(1, delay=5)
        client2_process = start_client(2, delay=8)
        
        print("\n✅ All processes started")
        print("\nMonitoring for 30 seconds...")
        print("(Watch for connection messages, training rounds, etc.)\n")
        
        # Monitor for 30 seconds
        end_time = time.time() + 30
        while time.time() < end_time:
            # Check if processes are still running
            if server_process.poll() is not None:
                print("\n⚠️  Server stopped")
                break
            if client1_process.poll() is not None:
                print("\n⚠️  Client 1 stopped")
            if client2_process.poll() is not None:
                print("\n⚠️  Client 2 stopped")
            
            time.sleep(1)
        
        print("\n" + "="*60)
        print("Test Complete!")
        print("="*60)
        print("\nIf you saw connection and training messages above, the setup is working!")
        print("\nTo run the full training:")
        print("1. Terminal 1: python server.py")
        print("2. Terminal 2: python client1.py localhost:8080")
        print("3. Terminal 3: python client2.py localhost:8080")
        
    except KeyboardInterrupt:
        print("\n\nStopping all processes...")
    finally:
        # Cleanup
        for name, proc in [("Server", server_process), ("Client 1", client1_process), ("Client 2", client2_process)]:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                    print(f"✅ {name} stopped")
                except:
                    try:
                        proc.kill()
                        print(f"⚠️  {name} force killed")
                    except:
                        pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

