import cv2
import sys
import os
import argparse
import numpy as np
import time
from ultralytics import YOLO

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser(description="Test YOLO-Pose for fast skeleton visualization.")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    args = parser.parse_args()

    # Determine source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Load YOLOv8n-Pose (pretrained, 17 COCO keypoints)
    print("Loading YOLOv8n-Pose model...")
    try:
        yolo_pose = YOLO('yolov8n-pose.pt')  # Will auto-download if not present
        print("✓ YOLOv8n-Pose loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO-Pose: {e}")
        return

    # Load Stick Detector
    print("Loading Stick Detector...")
    try:
        stick_model_path = os.path.join(project_root, 'deployment_package', 'weights', 'best.pt')
        if not os.path.exists(stick_model_path):
            stick_model_path = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
        
        stick_detector = YOLO(stick_model_path)
        print(f"✓ Stick Detector loaded from {stick_model_path}")
    except Exception as e:
        print(f"Warning: Could not load stick detector: {e}")
        stick_detector = None

    print("Press 'q' to quit.")
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0
    
    # COCO Keypoint names (17 keypoints)
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # COCO Skeleton connections
    COCO_SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        # Flip webcam for mirror feeling
        if source == 0:
            frame = cv2.flip(frame, 1)

        # YOLO-Pose Inference
        try:
            results = yolo_pose(frame, verbose=False)
            
            # Extract keypoints from first person detected
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.data  # Shape: [num_people, 17, 3] (x, y, conf)
                
                if len(keypoints) > 0:
                    kpts = keypoints[0].cpu().numpy()  # First person, 17 keypoints
                    
                    # Draw skeleton connections (lines)
                    for connection in COCO_SKELETON:
                        start_idx, end_idx = connection
                        start_pt = kpts[start_idx][:2]
                        end_pt = kpts[end_idx][:2]
                        start_conf = kpts[start_idx][2]
                        end_conf = kpts[end_idx][2]
                        
                        # Only draw if both keypoints are confident
                        if start_conf > 0.5 and end_conf > 0.5:
                            start_pt = (int(start_pt[0]), int(start_pt[1]))
                            end_pt = (int(end_pt[0]), int(end_pt[1]))
                            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)  # Green lines
                    
                    # Draw keypoints (circles)
                    for idx, kpt in enumerate(kpts):
                        x, y, conf = kpt
                        if conf > 0.5:  # Only draw confident keypoints
                            x, y = int(x), int(y)
                            cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)  # Magenta circles
                            # Draw keypoint index
                            cv2.putText(frame, str(idx), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # STICK DETECTION
                    if stick_detector is not None:
                        try:
                            stick_results = stick_detector(frame, verbose=False)
                            if len(stick_results) > 0 and stick_results[0].keypoints is not None:
                                stick_kpts = stick_results[0].keypoints.data
                                if len(stick_kpts) > 0:
                                    stick_kpt = stick_kpts[0].cpu().numpy()  # First stick
                                    grip = stick_kpt[0][:2]  # Grip point
                                    tip = stick_kpt[1][:2]   # Tip point
                                    grip_conf = stick_kpt[0][2]
                                    tip_conf = stick_kpt[1][2]
                                    
                                    if grip_conf > 0.3 and tip_conf > 0.3:
                                        grip_pt = (int(grip[0]), int(grip[1]))
                                        tip_pt = (int(tip[0]), int(tip[1]))
                                        
                                        # Adaptive Scaling Logic (same as test_stick_on_video.py)
                                        # COCO indices: 5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip
                                        l_sh = kpts[5][:2]
                                        r_sh = kpts[6][:2]
                                        l_hip = kpts[11][:2]
                                        r_hip = kpts[12][:2]
                                        
                                        shoulder_width = np.linalg.norm(l_sh - r_sh)
                                        torso_len = (np.linalg.norm(l_sh - l_hip) + np.linalg.norm(r_sh - r_hip)) / 2.0
                                        ratio = shoulder_width / (torso_len + 1e-6)
                                        
                                        # Calculate stick length
                                        if ratio > 0.45:
                                            # FRONT VIEW: Torso scaling
                                            stick_px = torso_len * 1.5
                                            mode = f"FRONT (R={ratio:.2f})"
                                        else:
                                            # SIDE VIEW: Forearm scaling (simplified - using torso as fallback)
                                            stick_px = torso_len * 1.5
                                            mode = f"SIDE (R={ratio:.2f})"
                                        
                                        # Project new tip
                                        grip_arr = np.array(grip_pt)
                                        tip_arr = np.array(tip_pt)
                                        yolo_vec = tip_arr - grip_arr
                                        y_len = np.linalg.norm(yolo_vec)
                                        
                                        if y_len > 1e-6:
                                            direction_unit = yolo_vec / y_len
                                            new_tip = grip_arr + (direction_unit * stick_px)
                                            corrected_tip = (int(new_tip[0]), int(new_tip[1]))
                                            
                                            # Draw stick
                                            cv2.line(frame, grip_pt, corrected_tip, (0, 255, 255), 4)  # Cyan stick
                                            cv2.circle(frame, grip_pt, 6, (255, 255, 0), -1)  # Yellow grip
                                            cv2.circle(frame, corrected_tip, 6, (255, 255, 0), -1)  # Yellow tip
                                            cv2.putText(frame, mode, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        except Exception as e:
                            pass  # Silently skip stick detection errors
                    
                    # Display info
                    cv2.putText(frame, "YOLO-Pose (17 keypoints)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"Processing error: {e}")
            cv2.putText(frame, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 10:
            fps_end_time = time.time()
            fps_display = fps_frame_count / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Resize for display if too big
        h, w = frame.shape[:2]
        if h > 800:
            scale = 800 / h
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        cv2.imshow("YOLO-Pose Test (Fast Mode)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal FPS: {fps_display:.1f}")

if __name__ == "__main__":
    main()
