
import subprocess
import os
import sys
import shutil

def build():
    print("--- Starting Build Process ---")
    
    # 1. Check for PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Error: PyInstaller not installed.")
        return

    # 2. Clean previous builds
    print("Cleaning previous builds...")
    if os.path.exists('build'):
        shutil.rmtree('build', ignore_errors=True)
    if os.path.exists('dist'):
        shutil.rmtree('dist', ignore_errors=True)

    # 3. Create a launcher wrapper (optional but helpful for catching errors)
    # Actually, let's rely on console=True in spec first.

    # 4. Run PyInstaller
    print("Running PyInstaller...")
    cmd = [sys.executable, '-m', 'PyInstaller', 'TuroArnis.spec', '--noconfirm']
    
    try:
        subprocess.check_call(cmd)
        print("\n--- Build Successful ---")
        print("Executable located at: dist/TuroArnis/TuroArnis.exe")
        print("Please run it from powerhshell to see any startup errors: .\\dist\\TuroArnis\\TuroArnis.exe")
    except subprocess.CalledProcessError as e:
        print(f"\n--- Build Failed ---")
        print(e)

if __name__ == "__main__":
    build()
