"""
Quick run script to start the AI Document Assistant.

This script starts both the backend server and the Gradio interface.

Make sure "Ollama serve &" command is done in WSL
"""
import time
import os
import subprocess
import threading
import sys
import shutil

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

def run_ollama_command(args):
    is_windows = os.name == 'nt'
    is_wsl = False
    if not is_windows:
        try:
            with open('/proc/version', 'r') as f:
                if 'Microsoft' in f.read():
                    is_wsl = True
        except FileNotFoundError:
            pass

    if is_windows:
        cmd = ['wsl', 'ollama'] + args
    else:
        cmd = ['ollama'] + args

    if not is_windows and shutil.which('ollama') is None:
        return None

    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return None

def main():
    """Start both backend and frontend."""
    print("🤖 AI Document Assistant - Starting Services")
    print("=" * 50)

    # Check if Ollama is available (Windows-compatible and WSL-aware)
    result = run_ollama_command(['list'])
    if result is None or result.returncode != 0:
        print("⚠️  Ollama not found. Please install Ollama first.")
        print("   Visit: https://ollama.ai")
        print("   For Windows: Download and install from https://ollama.ai/download/windows")
        print("\n   Continuing without Ollama - some features may not work.")
    else:
        models=['tinyllama','qwen2:1.5b','gemma:2b','mistral','llama3.2:1b']
        for m in models:
            if m not in result.stdout.lower(): 
                print(f"⚠️{m} model not found. Installing...")
                pull_result = run_ollama_command(['pull', 'mistral'])
                if pull_result is None or pull_result.returncode != 0:
                    print(f"⚠️  Failed to pull {m} model.")
                else:
                    print(f"✅  {m} model installed successfully.")
            else:
                print(f"✅  {m} model is already installed.")

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

#python run.py --server_name 0.0.0.0 --server_port 7860