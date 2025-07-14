"""
Start the application with proper localhost configuration for Windows.
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def start_backend():
    """Start backend on localhost."""
    print("🚀 Starting backend on localhost:8000...")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "backend.server:app", 
        "--host", "localhost", 
        "--port", "8000"
    ], env=env)

def start_frontend():
    """Start frontend on localhost."""
    print("🌐 Starting frontend on localhost:8501...")
    time.sleep(3)  # Wait for backend
    
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    # Create streamlit config directory
    os.makedirs(".streamlit", exist_ok=True)
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "interface/simple_ui.py",
        "--server.address", "localhost",
        "--server.port", "8501"
    ], env=env)

def open_browser():
    """Open browser after services start."""
    time.sleep(8)  # Wait for services to fully start
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:8501")

def main():
    """Start application with localhost URLs."""
    print("🤖 AI Document Assistant - Localhost Version")
    print("=" * 50)
    
    # Start browser opener in background
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start backend in background
    backend_thread = Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    print("⏱️  Services starting...")
    print("📱 Frontend URL: http://localhost:8501")
    print("🔧 Backend URL: http://localhost:8000")
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()