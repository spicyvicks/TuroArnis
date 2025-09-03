# import necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# initialize mediapipe drawing utilities and pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    # calculates the angle between three points
    # a, b, and c are lists containing x and y coordinates
    a = np.array(a) # first point
    b = np.array(b) # middle point (vertex)
    c = np.array(c) # end point
    
    # calculate the angle using arctan2 and convert to degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # ensure angle is between 0 and 180
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

# --- setup video capture ---
# use 0 for webcam
cap = cv2.VideoCapture(0)

# --- setup counter variables ---
counter = 0 
stage = None

# setup mediapipe instance with confidence thresholds
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break
        
        # --- processing ---
        # recolor image from bgr to rgb for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # make detection
        results = pose.process(image)
    
        # recolor back to bgr for rendering with opencv
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # --- landmark extraction and angle calculation ---
        try:
            # extract landmarks if a pose is detected
            landmarks = results.pose_landmarks.landmark
            
            # get coordinates for the left arm
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # visualize angle on the image
            cv2.putText(image, f"{angle:.2f}", 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- repetition counter logic ---
            if angle > 160: # threshold for "down" position
                stage = "down"
            if angle < 30 and stage == 'down': # threshold for "up" position
                stage = "up"
                counter += 1
                print(f"reps: {counter}")
                       
        except Exception as e:
            # pass if no landmarks are detected
            # print(f"error: {e}") # uncomment for debugging
            pass
        
        # --- render counter display ---
        # setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # display "reps" text
        cv2.putText(image, 'reps', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # display counter value
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # display "stage" text
        cv2.putText(image, 'stage', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # display current stage
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # --- render pose landmarks ---
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        # display the final image in a window
        cv2.imshow('mediapipe feed', image)

        # exit loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # release resources and close windows
    cap.release()
    cv2.destroyAllWindows()