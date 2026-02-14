import sys
import os
import cv2
import numpy as np
import mediapipe as mp

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.utils.resource_path import get_resource_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_keypoint_methods.py <path_to_image>")
        # Try to find a default image
        potential_defaults = ["ashly_left_chest.jpg", "indira_crown_thrust.jpg", "image.png"]
        input_image_path = None
        for p in potential_defaults:
            if os.path.exists(os.path.join(project_root, p)):
                input_image_path = os.path.join(project_root, p)
                print(f"No image provided. Using default: {input_image_path}")
                break
        if not input_image_path:
            return
    else:
        input_image_path = sys.argv[1]

    if not os.path.exists(input_image_path):
        print(f"Error: Image not found at {input_image_path}")
        return

    # Load image
    frame = cv2.imread(input_image_path)
    if frame is None:
        print(f"Error: Could not read image {input_image_path}")
        return

    h_img, w_img = frame.shape[:2]
    print(f"Loaded image: {input_image_path} ({w_img}x{h_img})")

    # Initialize PoseAnalyzer (Method 4 logic is inside)
    # Using the same path logic as PoseAnalyzer or specific script logic
    stick_model_path = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
    if not os.path.exists(stick_model_path):
        # Alternative paths
        alt_paths = [
            os.path.join(project_root, 'deployment_package', 'weights', 'best.pt'),
            os.path.join(project_root, 'app', 'models', 'best.pt')
        ]
        for p in alt_paths:
            if os.path.exists(p):
                stick_model_path = p
                break
    
    print(f"Initializing PoseAnalyzer with stick model: {stick_model_path}")
    analyzer = PoseAnalyzer(
        detection_interval=1,
        stick_model_path=stick_model_path,
        debug_stick=True
    )

    # Process frame
    print("Processing frame...")
    results = analyzer.process_frame(frame, skip_ml_inference=False, skip_stick_detection=False)

    if not results:
        print("No person detected in the image.")
        return

    # Take the first person detected
    data = results[0]
    landmarks_abs = data.get('landmarks_absolute')
    stick_endpoints = data.get('stick_endpoints')

    # MediaPipe drawing tools
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    def draw_skeleton(img, landmarks, connections, color=(200, 200, 200), thickness=2):
        if not landmarks:
            return
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1 = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                p2 = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                if p1 != (0, 0) and p2 != (0, 0): # Avoid drawing from invalid/unset points
                    cv2.line(img, p1, p2, color, thickness)

    # --- IMAGE 1: METHOD 4 STICK ---
    img_method4 = frame.copy()
    thickness_line = 6
    thickness_circle = 8
    
    if stick_endpoints:
        grip, tip = stick_endpoints
        # Draw stick
        cv2.line(img_method4, grip, tip, (0, 255, 0), thickness_line)
        cv2.circle(img_method4, grip, thickness_circle, (0, 255, 0), -1)
        cv2.circle(img_method4, tip, thickness_circle, (0, 0, 255), -1)
        cv2.putText(img_method4, "GRIP", (grip[0]+15, grip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_method4, "TIP (METHOD 4)", (tip[0]+15, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Draw faint skeleton for context using absolute coordinates
    if landmarks_abs:
        draw_skeleton(img_method4, landmarks_abs, mp_pose.POSE_CONNECTIONS, color=(255, 255, 255), thickness=2)

    cv2.imwrite("output_method4.jpg", img_method4)
    print("Saved output_method4.jpg")

    # --- IMAGE 2: MEDIAPIPE 33 ---
    img_mp33 = frame.copy()
    if landmarks_abs:
        # Draw connections manually with absolute coords
        draw_skeleton(img_mp33, landmarks_abs, mp_pose.POSE_CONNECTIONS, color=(255, 0, 255), thickness=4)
            
        for idx, pt in enumerate(landmarks_abs):
            cx, cy = int(pt[0]), int(pt[1])
            if cx == 0 and cy == 0: continue
            cv2.circle(img_mp33, (cx, cy), 6, (0, 255, 255), -1)
            cv2.putText(img_mp33, str(idx), (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite("output_mediapipe33.jpg", img_mp33)
    print("Saved output_mediapipe33.jpg")

    # --- IMAGE 3: GCN 35 COMBINED ---
    img_gcn35 = frame.copy()
    
    # Define GCN Edges from model_architecture.py
    GCN_EDGES = [
        (11, 12), (11, 23), (12, 24), (23, 24), 
        (11, 13), (13, 15), (12, 14), (14, 16), 
        (23, 25), (25, 27), (24, 26), (26, 28),
        (15, 33), (16, 33), (33, 34)
    ]
         
    # Draw GCN connections
    if landmarks_abs and len(landmarks_abs) >= 33:
        # All points (33 body + 2 stick if available)
        all_pts = []
        for pt in landmarks_abs:
            all_pts.append((int(pt[0]), int(pt[1])))
        
        if stick_endpoints:
            all_pts.append(stick_endpoints[0]) # 33
            all_pts.append(stick_endpoints[1]) # 34
            
        # Draw edges
        for start_idx, end_idx in GCN_EDGES:
            if start_idx < len(all_pts) and end_idx < len(all_pts):
                p1 = all_pts[start_idx]
                p2 = all_pts[end_idx]
                if p1 == (0,0) or p2 == (0,0): continue
                
                # Different color for stick connections
                color = (0, 255, 255) if (start_idx >= 33 or end_idx >= 33) else (0, 255, 0)
                cv2.line(img_gcn35, p1, p2, color, 6) # Increased thickness to 6

        # Draw Nodes and Labels
        for idx, pt in enumerate(all_pts):
            if pt == (0,0): continue
            color = (255, 0, 0) if idx < 33 else (0, 0, 255)
            cv2.circle(img_gcn35, pt, 8, color, -1) # Increased radius to 8
            cv2.putText(img_gcn35, str(idx), (pt[0]+12, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



    cv2.imwrite("output_gcn35.jpg", img_gcn35)
    print("Saved output_gcn35.jpg")

    print("\nVisualization complete. Created:")
    print("- output_method4.jpg (Adaptive Stick Correction)")
    print("- output_mediapipe33.jpg (Standard 33 Body Landmarks)")
    print("- output_gcn35.jpg (Combined 35-node GCN Graph)")

if __name__ == "__main__":
    main()
