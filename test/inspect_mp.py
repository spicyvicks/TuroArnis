
import mediapipe as mp
import os
import sys

print(f"MediaPipe Version: {mp.__version__}")
print(f"MediaPipe File: {mp.__file__}")
print(f"Dir(mp): {dir(mp)}")

# Check if 'tasks' exists (new API)
try:
    import mediapipe.tasks
    print("mediapipe.tasks import successful")
except ImportError as e:
    print(f"mediapipe.tasks import failed: {e}")

# Check site-packages content again
mp_path = os.path.dirname(mp.__file__)
print(f"Listing {mp_path}:")
try:
    print(os.listdir(mp_path))
except Exception as e:
    print(e)
