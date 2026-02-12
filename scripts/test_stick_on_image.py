import sys
import os
import cv2
import argparse
import numpy as np

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.utils.resource_path import get_resource_path

# Helper for extrapolation
def extrapolate_stick(grip, tip, extend_grip=0.0, extend_tip=0.1):
    grip = np.array(grip)
    tip = np.array(tip)
    
    # Vector from grip to tip
    vec = tip - grip
    length = np.linalg.norm(vec)
    
    if length < 1e-6:
        return grip, tip
        
    # Normalize vector
    unit_vec = vec / length
    
    # Extrapolate
    new_grip = grip - (unit_vec * (length * extend_grip))
    new_tip = tip + (unit_vec * (length * extend_tip))
    
    return tuple(new_grip.astype(int)), tuple(new_tip.astype(int))

# Helper for Body-Relative Scaling (Pseudo-3D)
def body_relative_stick(grip, tip, landmarks_abs):
    if not landmarks_abs:
        return None
    
    # Landmarks: 11=L_Shoulder, 12=R_Shoulder, 23=L_Hip, 24=R_Hip
    try:
        l_shoulder = np.array(landmarks_abs[11][:2])
        l_hip = np.array(landmarks_abs[23][:2])
        r_shoulder = np.array(landmarks_abs[12][:2])
        r_hip = np.array(landmarks_abs[24][:2])
        
        torso_L = np.linalg.norm(l_shoulder - l_hip)
        torso_R = np.linalg.norm(r_shoulder - r_hip)
        avg_torso = (torso_L + torso_R) / 2
        
        # Arnis stick is approx 28 inches. Torso is approx 18-20 inches. Ratio ~1.5
        target_stick_len = avg_torso * 2.0 
        
        grip = np.array(grip)
        tip = np.array(tip)
        vec = tip - grip
        current_len = np.linalg.norm(vec)
        
        if current_len < 1e-6: return None
        
        unit_vec = vec / current_len
        
        handle_len = 0 # User requested: REMOVE extension for grip
        blade_len = target_stick_len * 1.0 # Full length extends to tip? Or just scale?
        # Let's keep blade_len as full length from grip if we don't extend back.
        
        new_grip = grip # Start exactly at detected grip
        new_tip = grip + (unit_vec * target_stick_len) # Extend full length forward
        
        return tuple(new_grip.astype(int)), tuple(new_tip.astype(int))
        
    except IndexError:
        return None

def main():
    parser = argparse.ArgumentParser(description="Test stick detection on a single image.")
    # Make image_path optional
    parser.add_argument("image_path", nargs='?', help="Path to the input image")
    parser.add_argument("--output", default="debug_stick_output.jpg", help="Path to save the output image")
    args = parser.parse_args()

    # Determine image path
    target_path = args.image_path
    if not target_path:
        # Try to find a default image in the current directory
        potential_defaults = [
            "1 (1).png",
            "test_image.jpg",
            "image.png",
            "Copy of crown_thrust_tip is above head_left foot forward right foot back.jpg"
        ]
        for p in potential_defaults:
            if os.path.exists(p):
                target_path = p
                print(f"No image argument provided. Using default found: {target_path}")
                break
    
    if not target_path:
        print("Error: No image path provided and no default test images found.")
        print("Usage: python scripts/test_stick_on_image.py <path_to_image>")
        return

    # Load image
    if not os.path.exists(target_path):
        print(f"Error: Image not found at {target_path}")
        return

    frame = cv2.imread(target_path)
    if frame is None:
        print(f"Error: Could not read image {target_path}")
        return

    print(f"Loaded image: {target_path} ({frame.shape})")

    # Initialize PoseAnalyzer
    try:
        # Use absolute path relative to project root for reliability
        stick_model_path = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
        
        print(f"Loading stick model from: {stick_model_path}")
        if not os.path.exists(stick_model_path):
             print(f"WARNING: Stick model not found at {stick_model_path}")
             # Try alternatives
             alt_paths = [
                 os.path.join(project_root, 'deployment_package', 'weights', 'best.pt'),
                 os.path.join(project_root, 'app', 'models', 'best.pt'),
                 os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
             ]
             
             for p in alt_paths:
                 if os.path.exists(p):
                     print(f"Found alternative at: {p}")
                     stick_model_path = p
                     break
        
        analyzer = PoseAnalyzer(
            detection_interval=1, # Process every frame
            stick_model_path=stick_model_path,
            debug_stick=True
        )
        
        print(f"PoseAnalyzer initialized. Stick model loaded: {analyzer.stick_detector is not None}")

        print("Running inference...")
        
        # Enable ML inference to get Landmarks (needed for torso measurement)
        results = analyzer.process_frame(frame, skip_ml_inference=False, skip_stick_detection=False)
        
        if not results:
            print("No person detected.")
        else:
            print(f"Detected {len(results)} person(s).")
            for i, data in enumerate(results):
                person_id = data.get('id', i)
                print(f"\nPerson {person_id}:")
                
                # Check stick
                if 'stick_endpoints' in data and data['stick_endpoints']:
                    grip, tip = data['stick_endpoints']
                    print(f"  Stick DETECTED (Raw)!")
                    print(f"  Grip: {grip}, Tip: {tip}")
                    
                    # Draw Raw (Red/Blue detection)
                    cv2.line(frame, grip, tip, (0, 0, 255), 2) # Red line for raw
                    cv2.putText(frame, "RAW", (grip[0], grip[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Calculate Extrapolated
                    ext_grip, ext_tip = extrapolate_stick(grip, tip)
                    print(f"  Stick EXTRAPOLATED!")
                    print(f"  New Grip: {ext_grip}, New Tip: {ext_tip}")
                    
                    # Draw Extrapolated (Bright Cyan/Green)
                    cv2.line(frame, ext_grip, ext_tip, (255, 255, 0), 4) # Cyan thick line
                    cv2.circle(frame, ext_grip, 6, (0, 255, 0), -1) # Green new grip
                    cv2.circle(frame, ext_tip, 6, (0, 255, 0), -1) # Green new tip
                    cv2.putText(frame, "FULL", (ext_tip[0], ext_tip[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # 3. PSEUDO-3D / BODY SCALED (Magenta)
                    if 'landmarks_absolute' in data:
                        pseudo_res = body_relative_stick(grip, tip, data['landmarks_absolute'])
                        if pseudo_res:
                            p_grip, p_tip = pseudo_res
                            print(f"  Stick PSEUDO-3D Generated!")
                            # Offset more
                            off = 0
                            cv2.line(frame, (p_grip[0]+off, p_grip[1]), (p_tip[0]+off, p_tip[1]), (255, 0, 255), 4)
                            cv2.circle(frame, (p_grip[0]+off, p_grip[1]), 5, (255, 0, 255), -1)
                            cv2.putText(frame, "PSEUDO-3D", (p_tip[0]+off, p_tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    # 4. PnP 3D SOLVER (Yellow)
                    # Simple PnP approach:
                    # Known 3D points: Grip at (0,0,0), Tip at (0, 0, 28 inches)
                    # Detected 2D points: grip, tip
                    # Camera Matrix: Assume standard webcam (focal length ~ image width)
                    h, w, _ = frame.shape
                    focal_length = w  # Approx
                    center = (w/2, h/2)
                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype="double")
                    dist_coeffs = np.zeros((4,1)) # Assume no dist
                    
                    # 3D Model Points (Stick is 28 inches long)
                    # Let's use generic units. If 28 units = 28 inches.
                    model_points = np.array([
                        (0.0, 0.0, 0.0),        # Grip (Origin)
                        (0.0, 28.0, 0.0)        # Tip (Up along Y for now)
                    ])
                    
                    image_points = np.array([
                        grip,
                        tip
                    ], dtype="double")
                    
                    # solvePnP requires 4 points generally, or specialized flags for fewer.
                    # With 2 points, we can't fully solve rotation AND translation.
                    # Hack: Assume Grip is at fixed Z (e.g., 50 units away) and solve for Tip?
                    # Better Hack: Trigonometry.
                    
                    # 2D Length
                    vec = np.array(tip) - np.array(grip)
                    pixel_len = np.linalg.norm(vec)
                    
                    # If stick is flat to camera (Z=constant), pixel_len would be MAX.
                    # We need a reference for "Max Pixel Length" (when stick is straight up).
                    # Use Torso as reference again?
                    
                    if 'landmarks_absolute' in data:
                        # 1. Get scale (Pixels per Inch) from Torso
                        l_shoulder = np.array(data['landmarks_absolute'][11][:2])
                        l_hip = np.array(data['landmarks_absolute'][23][:2])
                        torso_px = np.linalg.norm(l_shoulder - l_hip)
                        # Torso is ~20 inches
                        px_per_inch = torso_px / 20.0
                        
                        expected_stick_px = 28.0 * px_per_inch
                        
                        # 2. Compare observed length to expected length
                        # observed = expected * cos(angle_away_from_camera)
                        # cos(angle) = observed / expected
                        ratio = min(1.0, pixel_len / expected_stick_px)
                        # angle = acos(ratio) implies tilt away from camera plane.
                        
                        # 3. Re-project
                        # If we know it's tilted, we can draw the "true" 3D vector projected?
                        # Actually, to visualize it, we'd just draw the... 2D line?
                        # Wait, the valid visualization of a 3D stick on a 2D image IS the 2D line.
                        # The user wants it to look "full length".
                        # So we Force the length to be expected_stick_px!
                        
                        # This ends up being identical to Method 3 (Body Relative Scaling), just derived via PnP logic.
                        # Let's draw it in Yellow to see if PnP logic yields different results than raw torso multiplier.
                        
                        unit_vec = vec / pixel_len
                        pnp_tip = np.array(grip) + (unit_vec * expected_stick_px)
                        
                        # Handle offset (grip is usually 4 inches up)
                        # stick_total = 28, handle = 4. ratio = 4/28 = ~0.14
                        grip_offset = 0.0 * expected_stick_px
                        tip_offset = 1.0 * expected_stick_px
                        
                        pnp_start = np.array(grip) - (unit_vec * grip_offset)
                        pnp_end = np.array(grip) + (unit_vec * tip_offset)
                        
                        print(f"  Stick PnP (Trig) Generated!")
                        off = 0
                        p1 = (int(pnp_start[0]+off), int(pnp_start[1]))
                        p2 = (int(pnp_end[0]+off), int(pnp_end[1]))
                        
                        cv2.line(frame, p1, p2, (0, 255, 255), 4) # Yellow
                        cv2.circle(frame, p1, 5, (0, 255, 255), -1)
                        cv2.putText(frame, "PnP-TRIG", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # 5. VECTOR PROJECTION (Green)
                    # Use MediaPipe World Landmarks (meters)
                    if 'world_landmarks' in data:
                        # 3D Landmarks: 13=L_Elbow, 15=L_Wrist | 14=R_Elbow, 16=R_Wrist
                        # Detect which hand holds the stick?
                        # Heuristic: Which wrist is closer to the Grip?
                        wl = data['world_landmarks'].landmark
                        lm_pixel = data['landmarks_absolute']
                        
                        l_wrist_px = np.array(lm_pixel[15][:2])
                        r_wrist_px = np.array(lm_pixel[16][:2])
                        
                        dist_l = np.linalg.norm(l_wrist_px - np.array(grip))
                        dist_r = np.linalg.norm(r_wrist_px - np.array(grip))
                        
                        # Indices for Elbow/Wrist
                        if dist_l < dist_r:
                            elb_idx, wrs_idx = 13, 15
                            side = "Left"
                        else:
                            elb_idx, wrs_idx = 14, 16
                            side = "Right"
                            
                        # Get 3D coords (x, y, z in meters)
                        # MediaPipe World coords: Origin is Hip Center.
                        elbow_3d = np.array([wl[elb_idx].x, wl[elb_idx].y, wl[elb_idx].z])
                        wrist_3d = np.array([wl[wrs_idx].x, wl[wrs_idx].y, wl[wrs_idx].z])
                        
                        # Forearm length in 3D (meters) - used ONLY for scaling
                        forearm_len_3d = np.linalg.norm(wrist_3d - elbow_3d)
                        
                        if forearm_len_3d > 1e-6:
                            # Stick Length = 0.71 meters (approx 28 inches)
                            stick_len_m = 0.71
                            
                            # Length ratio: how many "forearms" is one stick?
                            len_ratio = stick_len_m / forearm_len_3d
                            
                            # DIRECTION: Hybrid Body-Aware
                            # Vector A: Wrist (MP) -> Stick Tip (YOLO). This anchors the stick to the body.
                            # Vector B: Grip (YOLO) -> Stick Tip (YOLO). This is the raw detection.
                            # BLEND: Average them to stabilize jitter while keeping body context.
                            
                            # 1. Get Wrist (MediaPipe)
                            if side == "Left":
                                wrist_idx = 15
                            else:
                                wrist_idx = 16
                                
                            wrist_2d = np.array(lm_pixel[wrist_idx][:2])
                            
                            # 2. Keypoints from YOLO
                            grip_arr = np.array(grip)
                            tip_arr = np.array(tip)
                            
                            # Vector A: Wrist -> Tip (Body-Relative Direction)
                            vec_a = tip_arr - wrist_2d
                            
                            # Vector B: Grip -> Tip (Pure Detection Direction)
                            vec_b = tip_arr - grip_arr
                            
                            # Normalize
                            norm_a = np.linalg.norm(vec_a)
                            norm_b = np.linalg.norm(vec_b)
                            
                            final_unit = np.array([0.0, 0.0])
                            
                            if norm_a > 0 and norm_b > 0:
                                unit_a = vec_a / norm_a
                                unit_b = vec_b / norm_b
                                
                                # BLEND: 0% Body-Anchor + 100% Detection (Pure YOLO)
                                # User requested "GCN landmark graph" (Stick Model) instead of MediaPipe hand.
                                # Strategy: Trust the Stick Model (YOLO) for direction 100%.
                                #           Trust MediaPipe (3D Arm) for length/scale 100%.
                                blend_vec = unit_b 
                                final_unit = blend_vec / np.linalg.norm(blend_vec)
                                print(f"    Using PURE YOLO Direction (Stick Model).")
                                
                            elif norm_b > 0:
                                final_unit = vec_b / norm_b
                                print(f"    Wrist weak, using YOLO direction.")
                            else:
                                final_unit = np.array([0, -1])

                            # LENGTH: Adaptive Scaling (Front vs Side)
                            # User Logic: Front View -> Use Torso (Stable). Side View -> Use Forearm (Accurate).
                            
                            # Landmarks: 11=L_Shoulder, 12=R_Shoulder, 23=L_Hip, 24=R_Hip
                            l_sh = np.array(lm_pixel[11][:2])
                            r_sh = np.array(lm_pixel[12][:2])
                            l_hip = np.array(lm_pixel[23][:2])
                            r_hip = np.array(lm_pixel[24][:2])
                            
                            # Calculate Dimensions
                            shoulder_width = np.linalg.norm(l_sh - r_sh)
                            torso_len_l = np.linalg.norm(l_sh - l_hip)
                            torso_len_r = np.linalg.norm(r_sh - r_hip)
                            avg_torso_px = (torso_len_l + torso_len_r) / 2.0
                            
                            # Ratio: Shoulder Width / Torso Height
                            # Front view: Shoulders are wide (~0.8 - 1.0 ratio)
                            # Side view: Shoulders are narrow (< 0.5 ratio)
                            view_ratio = shoulder_width / (avg_torso_px + 1e-6)
                            
                            stick_px = 0
                            scale_mode = "Unknown"
                            
                            if view_ratio > 0.45:
                                # FRONT VIEW (Wide Shoulders) -> Use Torso Scaling
                                # Arm foreshortening is bad here, so trust the torso.
                                stick_px = avg_torso_px * 1.5
                                scale_mode = f"FRONT (Torso 1.5x, Ratio={view_ratio:.2f})"
                            else:
                                # SIDE VIEW (Narrow Shoulders) -> Use Forearm Scaling
                                # Arm length is visible and accurate here.
                                elbow_2d = np.array(lm_pixel[elb_idx][:2])
                                # wrist_2d defined above
                                forearm_px = np.linalg.norm(wrist_2d - elbow_2d)
                                stick_px = forearm_px * len_ratio # Uses 3D ratio
                                scale_mode = f"SIDE (Forearm 3D, Ratio={view_ratio:.2f})"
                            
                            # Draw from Grip along pure YOLO direction
                            proj_tip = np.array(grip) + (final_unit * stick_px)
                            
                            p_start = tuple(np.array(grip).astype(int))
                            p_end = (int(proj_tip[0]), int(proj_tip[1]))
                            
                            print(f"  Stick Vector-Proj (Green) Generated! Side: {side}")
                            print(f"    Mode: {scale_mode}, Stick px: {stick_px:.0f}")
                            
                            # Draw arrow
                            cv2.line(frame, p_start, p_end, (0, 255, 0), 4) # Green
                            cv2.circle(frame, p_start, 5, (0, 255, 0), -1)
                            cv2.putText(frame, "HYBRID-VEC", p_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                else:
                    print(f"  Stick NOT detected.")
                    if analyzer.stick_detector:
                        x1, y1, x2, y2 = data['bbox']
                        stick_res, _ = analyzer._detect_stick_with_yolo(frame, (x1, y1, x2, y2))
                        if stick_res:
                             grip, tip = stick_res
                             ext_grip, ext_tip = extrapolate_stick(grip, tip)
                             cv2.line(frame, ext_grip, ext_tip, (255, 255, 0), 4)
                             print(f"  [FORCE] Stick found on retry.")

        # Save output
        cv2.imwrite(args.output, frame)
        print(f"\nSaved debug output to: {args.output}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
