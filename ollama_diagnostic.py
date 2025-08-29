"""Diagnostic script to check Ollama setup and connection."""

#In WSL /usr/local/bin create setup_port_forward.sh
#echo "/usr/local/bin/setup_port_forward.sh" >> ~/.bashrc to autorun on WSl startup
#netstat -tulpn | grep 11434
#sudo lsof -i :11434
#sudo kill -9 [PID]
#netsh interface portproxy delete v4tov4 listenport=PORT_NUMBER listenaddress=127.0.0.1 

import requests
import json
import time
from langchain.llms import Ollama

OLLAMA_BASE_URL = "http://localhost:11434"  # WSL port-forwarded URL

def check_ollama_service():
    """Check if Ollama service is running and list available models."""
    print("🔍 Checking Ollama service...")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m['name'].split(':')[0] for m in data.get('models', [])]
        print(f"✅ Ollama service is running. Available models: {models}")
        return True, models
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama service: {e}")
        return False, []


def test_model_generate(model_name):
    """Test running a model using the correct /api/generate endpoint."""
    print(f"\n🔍 Testing model '{model_name}' via HTTP API...")
    payload = {
        "model": model_name,
        "prompt": "Hello, how are you?",
        "max_tokens": 50
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", 
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(payload),
                             timeout=20)
        if resp.status_code == 200:
            result = resp.json()
            output_text = result.get("output", [{}])[0].get("content", "")
            print(f"✅ Model responded successfully: {output_text[:100]}{'...' if len(output_text) > 100 else ''}")
            return True
        else:
            print(f"❌ Model call failed: HTTP {resp.status_code}, {resp.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling model: {e}")
        return False


def test_langchain_ollama(model_name):
    """Test LangChain Ollama wrapper."""
    print(f"\n🔍 Testing LangChain Ollama integration with '{model_name}'...")
    try:
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=model_name, timeout=30)
        start_time = time.time()
        resp = llm("Hello, how are you?")
        end_time = time.time()
        print(f"✅ LangChain integration works! Response: {resp[:100]}{'...' if len(resp) > 100 else ''}")
        print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        return True
    except Exception as e:
        print(f"❌ LangChain Ollama test failed: {e}")
        return False


def main():
    print("🚀 Ollama Diagnostic Script")
    print("="*50)

    service_ok, models = check_ollama_service()
    if not service_ok:
        print("❌ Ollama service is not running. Start it with 'ollama serve'.")
        return

    if not models:
        print("❌ No models available. Pull a model with 'ollama pull mistral'.")
        return

    # Use the first available model
    model_name = models[0]
    print(f"ℹ️  Using model: {model_name}")

    if not test_model_generate(model_name):
        print("❌ Model generate test failed. Check the model and Ollama service.")
        return

    if not test_langchain_ollama(model_name):
        print("❌ LangChain integration test failed.")
        return

    print("\n🎉 All checks passed! Ollama is ready for QA chain usage.")


if __name__ == "__main__":
    main()

