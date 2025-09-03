
import cv2
import mediapipe as mp
import numpy as np

# initialize 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

target_angles = {
    'left_elbow': 162.54,
    'left_shoulder': 82.82,
    'left_hip': 146.51,
    'left_knee': 150.30,
    'left_ankle': 107.31,
    'right_elbow': 124.84,
    'right_shoulder': 64.19,
    'right_hip': 177.95,
    'right_knee': 150.67,
    'right_ankle': 87.86
}

tolerance = 15.0 

video_path = 'videos/sample2.mp4' 
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('TuroArnis', cv2.WINDOW_NORMAL)
cv2.resizeWindow('TuroArnis', 1420, 800)

if not cap.isOpened():
    print(f"error: could not open video file: {video_path}")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            
            print("end of video reached.")
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        results = pose.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # left side
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
            l_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h]
            # right side
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
            r_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h]

            calculated_angles = {
                'left_elbow': calculate_angle(l_shoulder, l_elbow, l_wrist),
                'left_shoulder': calculate_angle(l_hip, l_shoulder, l_elbow),
                'left_hip': calculate_angle(l_shoulder, l_hip, l_knee),
                'left_knee': calculate_angle(l_hip, l_knee, l_ankle),
                'left_ankle': calculate_angle(l_knee, l_ankle, l_foot_index),
                'right_elbow': calculate_angle(r_shoulder, r_elbow, r_wrist),
                'right_shoulder': calculate_angle(r_hip, r_shoulder, r_elbow),
                'right_hip': calculate_angle(r_shoulder, r_hip, r_knee),
                'right_knee': calculate_angle(r_hip, r_knee, r_ankle),
                'right_ankle': calculate_angle(r_knee, r_ankle, r_foot_index)
            }
            
            # --- validate pose against target angles ---
            all_angles_correct = True
            for joint, target_angle in target_angles.items():
                calculated_angle = calculated_angles[joint]
                
                if not (target_angle - tolerance <= calculated_angle <= target_angle + tolerance):
                    all_angles_correct = False
                    break # if one angle is wrong, no need to check others
            
            
            if all_angles_correct:
                landmark_color = (0, 255, 0) 
                connection_color = (0, 255, 0)
            else:
                landmark_color = (0, 0, 255) 
                connection_color = (0, 0, 255)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2) 
                                     )               
                       
        except:
            
            pass
        
        
        cv2.imshow('TuroArnis', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()