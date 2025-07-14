"""
Fix dependency issues for Windows installation.
"""

import subprocess
import sys

def fix_dependencies():
    print("🔧 Fixing dependencies for Windows...")
    print("=" * 50)
    
    # Uninstall problematic packages
    print("\n1. Removing old packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "sentence-transformers", "huggingface-hub"])
    
    # Install specific compatible versions
    print("\n2. Installing compatible versions...")
    packages = [
        "huggingface-hub==0.17.3",
        "sentence-transformers==2.2.2",
        "transformers==4.35.2"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ Dependencies fixed!")
    print("\nYou can now run the application with:")
    print("   python run.py")
    print("   or")
    print("   run_windows.bat")

if __name__ == "__main__":
    fix_dependencies()