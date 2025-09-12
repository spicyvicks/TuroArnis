# import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import os 

# initialize mediapipe pose
mp_pose = mp.solutions.pose
# for single images, enable static_image_mode
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# initialize mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

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
    # add a small epsilon to avoid division by zero
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc + 1e-6)
    
    # ensure the value is in the valid range for arccos to avoid errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # calculate the angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# --- configuration ---
# specify the path to your image
image_path = 'labeled_form/left temple block.jpg' 
# specify the folder where processed images will be saved
output_folder = 'angle_images'

# --- setup ---
# create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"created directory: {output_folder}")

# read the image from the specified path
image = cv2.imread(image_path)

# check if the image was loaded successfully
if image is None:
    print(f"error: could not open or find the image at: {image_path}")
    exit()

# to improve performance, optionally mark the image as not writeable
image.flags.writeable = False
# convert the bgr image to rgb for mediapipe processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# process the image and detect the pose
results = pose.process(image_rgb)

# make the image writeable again to draw on it
image.flags.writeable = True

# create a list to hold the angle data for export
angle_data_for_export = []

# extract landmarks and calculate angles if a pose is detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape 

    # --- left side ---
    # get 3d coordinates
    l_shoulder_3d = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z * w]
    l_elbow_3d = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z * w]
    l_wrist_3d = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z * w]
    l_hip_3d = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z * w]
    l_knee_3d = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z * w]
    l_ankle_3d = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z * w]
    l_foot_index_3d = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z * w]

    # calculate 3d angles
    angle_l_elbow = calculate_angle_3d(l_shoulder_3d, l_elbow_3d, l_wrist_3d)
    angle_l_shoulder = calculate_angle_3d(l_hip_3d, l_shoulder_3d, l_elbow_3d)
    angle_l_hip = calculate_angle_3d(l_shoulder_3d, l_hip_3d, l_knee_3d)
    angle_l_knee = calculate_angle_3d(l_hip_3d, l_knee_3d, l_ankle_3d)
    angle_l_ankle = calculate_angle_3d(l_knee_3d, l_ankle_3d, l_foot_index_3d)
    
    # add to export list
    angle_data_for_export.extend([
        ("left elbow 3d", angle_l_elbow),
        ("left shoulder 3d", angle_l_shoulder),
        ("left hip 3d", angle_l_hip),
        ("left knee 3d", angle_l_knee),
        ("left ankle 3d", angle_l_ankle),
    ])

    # get 2d coordinates for visualization on the 2d image
    l_elbow_2d = [l_elbow_3d[0], l_elbow_3d[1]]
    l_shoulder_2d = [l_shoulder_3d[0], l_shoulder_3d[1]]
    l_hip_2d = [l_hip_3d[0], l_hip_3d[1]]
    l_knee_2d = [l_knee_3d[0], l_knee_3d[1]]
    l_ankle_2d = [l_ankle_3d[0], l_ankle_3d[1]]

    # visualize angles
    cv2.putText(image, f"{angle_l_elbow:.1f}", tuple(np.array(l_elbow_2d, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{angle_l_shoulder:.1f}", tuple(np.array(l_shoulder_2d, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # ... (add puttext for other joints if you want them displayed)

    # --- right side ---
    # get 3d coordinates
    r_shoulder_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z * w]
    r_elbow_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z * w]
    r_wrist_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z * w]
    r_hip_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z * w]
    r_knee_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z * w]
    r_ankle_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z * w]
    r_foot_index_3d = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z * w]
    
    # calculate 3d angles
    angle_r_elbow = calculate_angle_3d(r_shoulder_3d, r_elbow_3d, r_wrist_3d)
    angle_r_shoulder = calculate_angle_3d(r_hip_3d, r_shoulder_3d, r_elbow_3d)
    angle_r_hip = calculate_angle_3d(r_shoulder_3d, r_hip_3d, r_knee_3d)
    angle_r_knee = calculate_angle_3d(r_hip_3d, r_knee_3d, r_ankle_3d)
    angle_r_ankle = calculate_angle_3d(r_knee_3d, r_ankle_3d, r_foot_index_3d)
    
    # add to export list
    angle_data_for_export.extend([
        ("right elbow 3d", angle_r_elbow),
        ("right shoulder 3d", angle_r_shoulder),
        ("right hip 3d", angle_r_hip),
        ("right knee 3d", angle_r_knee),
        ("right ankle 3d", angle_r_ankle),
    ])

    # get 2d coordinates for visualization on the 2d image
    r_elbow_2d = [r_elbow_3d[0], r_elbow_3d[1]]
    
    # visualize angle
    cv2.putText(image, f"{angle_r_elbow:.1f}", tuple(np.array(r_elbow_2d, dtype=int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # ... (add puttext for other joints if you want them displayed)

    # draw pose landmarks
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

    # --- export results ---
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. save processed image
    output_image_filename = f"{base_filename}_processed.jpg"
    output_image_path = os.path.join(output_folder, output_image_filename)
    cv2.imwrite(output_image_path, image)
    print(f"[info] processed image saved to: {output_image_path}")

    # 2. save 3d angles to a text file
    output_txt_path = f"{base_filename}_angles_3d.txt"
    with open(output_txt_path, 'w') as f:
        f.write(f"3d pose angles for {os.path.basename(image_path)}\n")
        f.write("-----------------------------------------\n")
        for name, angle in angle_data_for_export:
            f.write(f"{name}: {angle:.2f} degrees\n")
    print(f"[info] 3d angle data successfully exported to: {output_txt_path}")
    
else:
    # if no pose is detected
    print("[info] no pose landmarks detected in the image. no files were created.")

# display the final image
cv2.imshow('3d angle detection', image)
cv2.waitKey(0) # waits for a key press to close

# clean up
cv2.destroyAllWindows()
pose.close()