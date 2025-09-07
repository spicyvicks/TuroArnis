# import necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# initialize mediapipe drawing utilities and pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle_3d(a, b, c):
    # calculates the angle between three 3d points
    # a, b, and c are lists or tuples with [x, y, z] coordinates
    a = np.array(a)  # first point
    b = np.array(b)  # middle point (the vertex of the angle)
    c = np.array(c)  # end point
    
    # create vectors from the points
    ba = a - b
    bc = c - b
    
    # calculate the dot product of the vectors
    dot_product = np.dot(ba, bc)
    
    # calculate the magnitude (length) of each vector
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    # calculate the cosine of the angle using the dot product formula
    # add a small epsilon (1e-6) to avoid division by zero
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc + 1e-6)
    
    # ensure the value is in the valid range for arccos to avoid math errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # calculate the angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# --- configuration for target pose ---
# these targets now represent the desired 3d angles
target_angles = {
    'left_elbow': 132.30 ,
    'left_shoulder': 106.78,
    'left_hip': 109.32,
    'left_knee': 157.97,
    'left_ankle': 57.37,
    'right_elbow': 148.90,
    'right_shoulder': 80.17,
    'right_hip': 120.76,
    'right_knee': 161.05,
    'right_ankle': 137.75
}
# define the tolerance (margin of error) in degrees
tolerance = 15.0 

# --- setup video capture ---
video_path = 'videos/sample2.mp4' 
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('turoarnis', cv2.WINDOW_NORMAL)
cv2.resizeWindow('turoarnis', 1420, 800)

if not cap.isOpened():
    print(f"error: could not open video file: {video_path}")
    exit()

# setup mediapipe instance with confidence thresholds
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            print("end of video reached.")
            break
        
        # recolor image from bgr to rgb for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # make detection
        results = pose.process(image)
    
        # recolor back to bgr for rendering with opencv
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # --- get 3d coordinates for all joints ---
            # left side
            l_shoulder_3d = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z * w]
            l_elbow_3d = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z * w]
            l_wrist_3d = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z * w]
            l_hip_3d = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z * w]
            l_knee_3d = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z * w]
            l_ankle_3d = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z * w]
            l_foot_index_3d = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z * w]
            # right side
            r_shoulder_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z * w]
            r_elbow_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z * w]
            r_wrist_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z * w]
            r_hip_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z * w]
            r_knee_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z * w]
            r_ankle_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z * w]
            r_foot_index_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z * w]

            # calculate all 3d angles
            calculated_angles = {
                'left_elbow': calculate_angle_3d(l_shoulder_3d, l_elbow_3d, l_wrist_3d),
                'left_shoulder': calculate_angle_3d(l_hip_3d, l_shoulder_3d, l_elbow_3d),
                'left_hip': calculate_angle_3d(l_shoulder_3d, l_hip_3d, l_knee_3d),
                'left_knee': calculate_angle_3d(l_hip_3d, l_knee_3d, l_ankle_3d),
                'left_ankle': calculate_angle_3d(l_knee_3d, l_ankle_3d, l_foot_index_3d),
                'right_elbow': calculate_angle_3d(r_shoulder_3d, r_elbow_3d, r_wrist_3d),
                'right_shoulder': calculate_angle_3d(r_hip_3d, r_shoulder_3d, r_elbow_3d),
                'right_hip': calculate_angle_3d(r_shoulder_3d, r_hip_3d, r_knee_3d),
                'right_knee': calculate_angle_3d(r_hip_3d, r_knee_3d, r_ankle_3d),
                'right_ankle': calculate_angle_3d(r_knee_3d, r_ankle_3d, r_foot_index_3d)
            }
            
            # validate pose against target angles
            all_angles_correct = True
            for joint, target_angle in target_angles.items():
                calculated_angle = calculated_angles[joint]
                
                if not (target_angle - tolerance <= calculated_angle <= target_angle + tolerance):
                    all_angles_correct = False
                    break # if one angle is wrong, no need to check others
            
            # set feedback color based on validation
            if all_angles_correct:
                landmark_color = (0, 255, 0) # green for correct
                connection_color = (0, 255, 0)
            else:
                landmark_color = (0, 0, 255) # red for incorrect
                connection_color = (0, 0, 255)

            # render pose landmarks with feedback color
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2) 
                                     )               
                       
        except:
            # pass if no landmarks are detected
            pass
        
        # display the final image in the window
        cv2.imshow('turoarnis', image)

        # exit loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # release resources and close windows
    cap.release()
    cv2.destroyAllWindows()