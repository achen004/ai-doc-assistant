"""
Quick run script to start the AI Document Assistant.

This script starts both the backend server and the Gradio interface.
"""

import subprocess
import sys
import time
import threading
import os

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI backend server...")
    # Set Python path for Windows
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "backend.server:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ], env=env)

def start_frontend():
    """Start the Gradio frontend."""
    print("🌐 Starting Gradio frontend...")
    # Wait a moment for backend to start
    time.sleep(3)
    # Set Python path for Windows
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    subprocess.run([sys.executable, "interface/ui.py"], env=env)

def main():
    """Start both backend and frontend."""
    print("🤖 AI Document Assistant - Starting Services")
    print("=" * 50)
    
    # Check if Ollama is available (Windows-compatible)
    try:
        # Try to find ollama executable
        ollama_cmd = "ollama.exe" if os.name == 'nt' else "ollama"
        result = subprocess.run([ollama_cmd, "list"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            if "mistral" not in result.stdout:
                print("⚠️  Mistral model not found. Installing...")
                subprocess.run([ollama_cmd, "pull", "mistral"], shell=True)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  Ollama not found. Please install Ollama first.")
        print("   Visit: https://ollama.ai")
        print("   For Windows: Download and install from https://ollama.ai/download/windows")
        print("\n   Continuing without Ollama - some features may not work.")
        # Don't return - let the user continue without Ollama
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()