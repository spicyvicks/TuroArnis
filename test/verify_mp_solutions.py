
import mediapipe as mp
try:
    print(f"MediaPipe Version: {mp.__version__}")
    mp_pose = mp.solutions.pose
    print("Default pose solution found")
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    print("Pose object created successfully")
    print("✅ MediaPipe Solutions API is Working!")
except Exception as e:
    print(f"❌ Verification failed: {e}")
