import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# --- CHOOSE YOUR VIDEO SOURCE ---
# To use a webcam:
# video_path = 0

# To use a video file:
video_path = 'videos/sample1.mp4' # <-- IMPORTANT: REPLACE WITH YOUR VIDEO FILE PATH

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video source: {video_path}")
    exit()

while cap.isOpened():
    # Read a frame from the video
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame or end of video.")
        # If loading a video, you can use 'break' instead of 'continue'.
        break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose
    results = pose.process(image_rgb)

    # Revert the image back to BGR and make it writeable
    image.flags.writeable = True
    # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Not strictly necessary as we draw on the original 'image'

    # Draw the pose annotation on the image.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # You can also access individual landmarks like this:
        # landmarks = results.pose_landmarks.landmark
        # nose_landmark = landmarks[mp_pose.PoseLandmark.NOSE]
        # print(f"Nose coordinates: (x: {nose_landmark.x}, y: {nose_landmark.y}, z: {nose_landmark.z})")


    # Display the resulting frame
    cv2.imshow('MediaPipe Pose Detection', image)

    # Exit if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Don't forget to close the pose model
pose.close()