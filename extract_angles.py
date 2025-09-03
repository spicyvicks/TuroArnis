import cv2
import mediapipe as mp
import numpy as np
import os 

# initialize
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# initialize 
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    # calculates the angle between three points 
    # [x, y] coordinates
    a = np.array(a)  # first point
    b = np.array(b)  # vertex point
    c = np.array(c)  # end point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- path ---
# to img
image_path = 'labeled_form/left temple strike.jpg' 

# output for angle img
output_folder = 'angle_images'

# --- if folder doesn't exist ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"created directory: {output_folder}")

# read img
image = cv2.imread(image_path)

# check img
if image is None:
    print(f"error: could not open or find the image at: {image_path}")
    exit()

# mark img not writeable
image.flags.writeable = False
# convert bgr to rgb
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect pose
results = pose.process(image_rgb)

# writeable img
image.flags.writeable = True

# angle list
angle_data_for_export = []

# extract landmarks & calculate angles 
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape 

    # --- left side ---
    # coordinates
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
    l_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h]

    # angles
    angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
    angle_l_shoulder = calculate_angle(l_hip, l_shoulder, l_elbow)
    angle_l_hip = calculate_angle(l_shoulder, l_hip, l_knee)
    angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)
    angle_l_ankle = calculate_angle(l_knee, l_ankle, l_foot_index)
    
    # export list
    angle_data_for_export.extend([
        ("left elbow", angle_l_elbow),
        ("left shoulder", angle_l_shoulder),
        ("left hip", angle_l_hip),
        ("left knee", angle_l_knee),
        ("left ankle", angle_l_ankle),
    ])

    # angle visual
    cv2.putText(image, f"{angle_l_elbow:.1f}", tuple(np.array(l_elbow, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_l_shoulder:.1f}", tuple(np.array(l_shoulder, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_l_hip:.1f}", tuple(np.array(l_hip, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_l_knee:.1f}", tuple(np.array(l_knee, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_l_ankle:.1f}", tuple(np.array(l_ankle, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # --- right side ---
    # coordinates
    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
    r_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h]

    # calculate
    angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
    angle_r_shoulder = calculate_angle(r_hip, r_shoulder, r_elbow)
    angle_r_hip = calculate_angle(r_shoulder, r_hip, r_knee)
    angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)
    angle_r_ankle = calculate_angle(r_knee, r_ankle, r_foot_index)
    
    # export list
    angle_data_for_export.extend([
        ("right elbow", angle_r_elbow),
        ("right shoulder", angle_r_shoulder),
        ("right hip", angle_r_hip),
        ("right knee", angle_r_knee),
        ("right ankle", angle_r_ankle),
    ])

    # visual
    cv2.putText(image, f"{angle_r_elbow:.1f}", tuple(np.array(r_elbow, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_r_shoulder:.1f}", tuple(np.array(r_shoulder, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_r_hip:.1f}", tuple(np.array(r_hip, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_r_knee:.1f}", tuple(np.array(r_knee, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_r_ankle:.1f}", tuple(np.array(r_ankle, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # draw
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

    # --- export results ---
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. save 
    output_image_filename = f"{base_filename}_processed.jpg"
    output_image_path = os.path.join(output_folder, output_image_filename)
    cv2.imwrite(output_image_path, image)
    print(f"[info] processed image saved to: {output_image_path}")

    # 2. save txt
    output_txt_path = f"{base_filename}_angles.txt"
    with open(output_txt_path, 'w') as f:
        f.write(f"pose angles for {os.path.basename(image_path)}\n")
        f.write("-----------------------------------------\n")
        for name, angle in angle_data_for_export:
            f.write(f"{name}: {angle:.2f} degrees\n")
    print(f"[info] angle data successfully exported to: {output_txt_path}")
    
else:
    # if no pose
    print("[info] no pose landmarks detected in the image. no files were created.")

# display
cv2.imshow('Angle Detection', image)

cv2.waitKey(0)

cv2.destroyAllWindows()
pose.close()