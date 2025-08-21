"""Diagnostic script to check Ollama setup and connection."""

#In WSL /usr/local/bin create setup_port_forward.sh
#echo "/usr/local/bin/setup_port_forward.sh" >> ~/.bashrc to autorun on WSl startup
#netstat -tulpn | grep 11434
#sudo lsof -i :11434
#sudo kill -9 [PID]
#netsh interface portproxy delete v4tov4 listenport=PORT_NUMBER listenaddress=127.0.0.1 

import requests
import json
from langchain.llms import Ollama
import time

def check_ollama_service():
    """Check if Ollama service is running."""
    print("🔍 Checking Ollama service...")
    
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            
            data = response.json()
            models = data.get('models', [])
            print(f"📦 Available models: {[model['name'] for model in models]}")
            
            return True, models
        else:
            print(f"❌ Ollama service responded with status code: {response.status_code}")
            return False, []
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama service at http://127.0.0.1:11434")
        print("💡 Make sure Ollama is running: ollama serve")
        return False, []
    
    except requests.exceptions.Timeout:
        print("❌ Ollama service is not responding (timeout)")
        return False, []
    
    except Exception as e:
        print(f"❌ Unexpected error checking Ollama: {str(e)}")
        return False, []


def test_model_availability(model_name="mistral"):
    """Test if a specific model is available."""
    print(f"\n🔍 Testing model availability: {model_name}")
    
    try:
        # Try to get model info
        response = requests.get(f"http://127.0.0.1:11434/api/show", 
                              json={"name": model_name}, 
                              timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Model '{model_name}' is available")
            return True
        else:
            print(f"❌ Model '{model_name}' not found")
            print(f"💡 Try running: ollama pull {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking model: {str(e)}")
        return False


def test_langchain_ollama(model_name="mistral"):
    """Test LangChain Ollama integration."""
    print(f"\n🔍 Testing LangChain Ollama integration with {model_name}...")
    
    try:
        # Create Ollama instance
        llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model=model_name,
            timeout=30  # Increase timeout
        )
        
        # Test simple query
        print("🧪 Testing simple query...")
        start_time = time.time()
        response = llm("Hello, how are you? Please respond briefly.")
        end_time = time.time()
        
        print(f"✅ LangChain integration working!")
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain Ollama test failed: {str(e)}")
        
        # Check if it's a connection issue
        if "Connection" in str(e) or "connection" in str(e).lower():
            print("💡 This looks like a connection issue. Check if:")
            print("   - Ollama is running: ollama serve")
            print("   - No firewall blocking port 11434")
            print("   - Model is loaded: ollama run mistral")
        
        return False


def test_model_performance(model_name="mistral"):
    """Test model performance with longer query."""
    print(f"\n🔍 Testing model performance with longer query...")
    
    try:
        llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model=model_name,
            timeout=60  # Longer timeout for complex queries
        )
        
        test_query = """
        Based on this context: "The weather is sunny and warm today. 
        People are enjoying outdoor activities in the park."
        
        Question: What is the weather like?
        """
        
        print("🧪 Testing longer query with context...")
        start_time = time.time()
        response = llm(test_query)
        end_time = time.time()
        
        print(f"✅ Performance test passed!")
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False


def main():
    """Run all diagnostic tests."""
    print("🚀 Ollama Diagnostic Script")
    print("=" * 50)
    
    # Step 1: Check service
    service_ok, models = check_ollama_service()
    
    if not service_ok:
        print("\n❌ Cannot proceed - Ollama service is not running")
        print("\n🔧 To fix:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Start the service: ollama serve")
        print("3. Pull a model: ollama pull mistral")
        return
    
    # Step 2: Check model
    model_name = "mistral"
    if models:
        # Use first available model if mistral not found
        available_model_names = [model['name'].split(':')[0] for model in models]
        if model_name not in available_model_names:
            model_name = available_model_names[0]
            print(f"ℹ️  Using available model: {model_name}")
    
    model_ok = test_model_availability(model_name)
    
    if not model_ok:
        print(f"\n❌ Model '{model_name}' not available")
        print(f"🔧 To fix: ollama pull {model_name}")
        return
    
    # Step 3: Test LangChain integration
    langchain_ok = test_langchain_ollama(model_name)
    
    if not langchain_ok:
        return
    
    # Step 4: Test performance
    perf_ok = test_model_performance(model_name)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Ollama Service: {'✅' if service_ok else '❌'}")
    print(f"Model Available: {'✅' if model_ok else '❌'}")
    print(f"LangChain Integration: {'✅' if langchain_ok else '❌'}")
    print(f"Performance Test: {'✅' if perf_ok else '❌'}")
    
    if all([service_ok, model_ok, langchain_ok, perf_ok]):
        print("\n🎉 All tests passed! Your Ollama setup should work with the QA chain.")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")


if __name__ == "__main__":
    main()