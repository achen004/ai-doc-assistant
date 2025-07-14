"""
Install simplified dependencies for Windows.
"""

import subprocess
import sys

def install_simple():
    print("🔧 Installing simplified dependencies for Windows...")
    print("=" * 60)
    
    # Core packages that are known to work on Windows
    packages = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0", 
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "streamlit==1.28.1",
        "requests==2.31.0",
        "numpy==1.24.3",
        "pillow==10.1.0",
        "PyMuPDF==1.23.8",
        # Skip ML packages for now - install manually if needed
    ]
    
    print("\n📦 Installing packages...")
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}")
            print(f"      Error: {e}")
    
    print("\n🤖 Optional ML packages (may have compatibility issues):")
    print("   If you want full AI functionality, try installing these manually:")
    print("   pip install sentence-transformers==2.2.2")
    print("   pip install transformers==4.35.2") 
    print("   pip install torch")
    print("   pip install faiss-cpu")
    
    print("\n✅ Basic installation complete!")
    print("\n🚀 You can now start the application:")
    print("   run_windows_simple.bat")
    print("   or")
    print("   python run_simple.py")

if __name__ == "__main__":
    install_simple()