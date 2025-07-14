"""
Simple launcher for Windows - uses Streamlit instead of Gradio.
"""

import subprocess
import sys
import time
import threading
import os

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI backend server...")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "backend.server:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ], env=env)

def start_frontend():
    """Start the Streamlit frontend."""
    print("🌐 Starting Streamlit frontend...")
    time.sleep(3)  # Wait for backend
    
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "interface/simple_ui.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ], env=env)

def main():
    """Start both backend and frontend."""
    print("🤖 AI Document Assistant - Simple Version")
    print("=" * 50)
    
    # Check if port 8000 is already in use
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            print("⚠️  Port 8000 is already in use.")
            print("   Please stop any running instances or use a different port.")
            print("   You can manually start the frontend with:")
            print("   streamlit run interface/simple_ui.py")
            return
            
    except Exception:
        pass
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()