
import subprocess
import time
import os
import sys

def test_dist():
    exe_path = os.path.abspath("dist/TuroArnis/TuroArnis.exe")
    if not os.path.exists(exe_path):
        print(f"Executable not found at {exe_path}")
        return

    print(f"Launching {exe_path}...")
    
    # Start the process
    # varying creationflags might be needed to hide/show window, but since console=True in spec, it should just show.
    process = subprocess.Popen(
        [exe_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE # Try to let it have its own console
    )

    print(f"Process started with PID {process.pid}. Waiting 10 seconds...")
    
    try:
        # Wait for 10 seconds to see if it crashes immediately
        # If it's a GUI app, it should theoretically run forever until closed.
        # If it crashes, process.poll() will return a code.
        
        for i in range(10):
            time.sleep(1)
            ret = process.poll()
            if ret is not None:
                print(f"Process exited early with return code {ret}!")
                # Capture output if possible (might be tricky if it went to its own console)
                # But we piped stdout/stderr, so maybe we get it.
                stdout, stderr = process.communicate()
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
                return

            print(f"T+{i+1}s: Still running...")

        print("Success! Process ran for 10 seconds without crashing.")
        print("Terminating process...")
        process.terminate()
        try:
           process.wait(timeout=5)
        except subprocess.TimeoutExpired:
           process.kill()
           
    except Exception as e:
        print(f"Error during test: {e}")
        if process.poll() is None:
             process.kill()

if __name__ == "__main__":
    test_dist()
