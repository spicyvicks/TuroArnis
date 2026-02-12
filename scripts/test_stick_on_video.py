import cv2
import sys
import os
import argparse
import numpy as np
import mediapipe as mp
import torch
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer

def extrapolate_stick(grip, tip):
    # Same helper as before
    grip = np.array(grip)
    tip = np.array(tip)
    vec = tip - grip
    length = np.linalg.norm(vec)
    if length < 1e-6: return grip, tip
    unit_vec = vec / length
    # Extend backwards 30%, forwards 10%
    new_grip = grip - (unit_vec * length * 0.3)
    new_tip = tip + (unit_vec * length * 0.1)
    return tuple(new_grip.astype(int)), tuple(new_tip.astype(int))

def main():
    parser = argparse.ArgumentParser(description="Test stick method on video/webcam.")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--interval", type=int, default=3, help="Detection interval (default: 3)")
    args = parser.parse_args()

    # Determine source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Initialize Analyzer
    try:
        stick_model_path = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
        if not os.path.exists(stick_model_path):
             alt_paths = [
                 os.path.join(project_root, 'deployment_package', 'weights', 'best.pt'),
                 os.path.join(project_root, 'app', 'models', 'best.pt')
             ]
             for p in alt_paths:
                 if os.path.exists(p):
                     stick_model_path = p
                     break
        
        print(f"Loading stick model: {stick_model_path}")
        print(f"Detection interval: {args.interval} (YOLO runs every {args.interval} frames)")
        analyzer = PoseAnalyzer(
            detection_interval=args.interval,  # Run YOLO every N frames
            stick_model_path=stick_model_path,
            debug_stick=False # Turn off internal print spam
        )
    except Exception as e:
        print(f"Failed to init analyzer: {e}")
        return

    print("Press 'q' to quit.")
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        # Flip webcam for mirror feeling
        if source == 0:
            frame = cv2.flip(frame, 1)

        # OPTIMIZATION: Resize input frame for faster processing
        h_orig, w_orig = frame.shape[:2]
        target_width = 1280
        if w_orig > target_width:
            scale = target_width / w_orig
            frame = cv2.resize(frame, (target_width, int(h_orig * scale)))

        # Process Frame
        # Analyzer internally calls stick detection
        try:
            results = analyzer.process_frame(frame, skip_ml_inference=False, skip_stick_detection=False)
        except Exception as e:
            print(f"Processing error: {e}")
            results = []

        if results:
            for data in results:
                # 1. Draw GCN Body Keypoints (Nodes 0-32: MediaPipe Body)
                if 'landmarks_absolute' in data and data['landmarks_absolute']:
                    lm_pixel = data['landmarks_absolute']
                    
                    # First, draw skeleton connections (lines)
                    mp_pose = mp.solutions.pose
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(lm_pixel) and end_idx < len(lm_pixel):
                            start_pt = (int(lm_pixel[start_idx][0]), int(lm_pixel[start_idx][1]))
                            end_pt = (int(lm_pixel[end_idx][0]), int(lm_pixel[end_idx][1]))
                            cv2.line(frame, start_pt, end_pt, (245, 117, 66), 2)  # Orange lines
                    
                    # Then, draw all 33 MediaPipe body landmarks as circles on top
                    for idx, lm in enumerate(lm_pixel):
                        x, y = int(lm[0]), int(lm[1])
                        # Draw circle for each keypoint
                        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)  # Yellow circles
                        # Draw index number
                        cv2.putText(frame, str(idx), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # 2. Draw Stick Keypoints (Nodes 33-34: YOLO Stick)
                if 'stick_endpoints' in data and data['stick_endpoints']:
                    grip, tip = data['stick_endpoints']
                    
                    # Draw stick keypoints as larger circles
                    cv2.circle(frame, grip, 6, (255, 0, 255), -1)  # Magenta for Grip (Node 33)
                    cv2.circle(frame, tip, 6, (255, 0, 255), -1)   # Magenta for Tip (Node 34)
                    cv2.putText(frame, "33", (grip[0]+5, grip[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, "34", (tip[0]+5, tip[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Draw stick line in GREEN (Final Result)
                    cv2.line(frame, grip, tip, (0, 255, 0), 4)
                    cv2.putText(frame, "FINAL (Adaptive)", tip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Show adaptive mode info
                    if 'landmarks_absolute' in data:
                        lm_pixel = data['landmarks_absolute']
                        
                        l_sh = np.array(lm_pixel[11][:2])
                        r_sh = np.array(lm_pixel[12][:2])
                        l_hip = np.array(lm_pixel[23][:2])
                        r_hip = np.array(lm_pixel[24][:2])
                        
                        shoulder_width = np.linalg.norm(l_sh - r_sh)
                        torso_len = (np.linalg.norm(l_sh - l_hip) + np.linalg.norm(r_sh - r_hip)) / 2.0
                        ratio = shoulder_width / (torso_len + 1e-6)
                        
                        mode = "FRONT (Torso)" if ratio > 0.45 else "SIDE (Forearm)"
                        cv2.putText(frame, f"Mode: {mode} (R={ratio:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, "GCN Nodes: 35 (33 Body + 2 Stick)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "No Stick", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 10:
            fps_end_time = time.time()
            fps_display = fps_frame_count / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Resize for display if too big
        h, w = frame.shape[:2]
        if h > 800:
            scale = 800 / h
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        cv2.imshow("Adaptive Stick Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    analyzer.close()

if __name__ == "__main__":
    main()
