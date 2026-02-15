"""
Image-based testing version of the Kiosk app
Loads a static image instead of camera feed for faster testing
Now uses pure recognition mode (no target pose comparison)
NEW: Supports multi-user zoning mode
"""
import cv2
import numpy as np
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.utils.resource_path import get_resource_path

# Confidence thresholds (same as main app)
CONFIDENCE_THRESHOLDS = {
    'front': 0.35,  
    'left': 0.35,   
    'right': 0.35  
}

def test_multi_user_classification(image_paths, viewpoints, user_names=None):
    """
    Test GCN classification with multi-user zoning (like main kiosk app)
    
    Args:
        image_paths: List of paths to test images (one per user) OR single image path
        viewpoints: List of viewpoints for each user ('front', 'left', 'right')
        user_names: Optional list of names for each user
    """
    # Check if single image with multiple zones
    if len(image_paths) == 1 and len(viewpoints) > 1:
        # Single image, split into zones
        return test_single_image_multi_zone(image_paths[0], viewpoints, user_names)
    
    num_users = len(image_paths)
    
    if num_users < 1 or num_users > 3:
        print("[✗] Number of users must be between 1 and 3")
        return
    
    if len(viewpoints) != num_users:
        print("[✗] Number of viewpoints must match number of users")
        return
    
    if user_names is None:
        user_names = [f"User {i+1}" for i in range(num_users)]
    
    print(f"\n{'='*60}")
    print(f"Multi-User Image Classification Test (Zoning Mode)")
    print(f"{'='*60}")
    print(f"Number of Users: {num_users}")
    for i in range(num_users):
        print(f"  Zone {i+1}: {user_names[i]} - {image_paths[i]} ({viewpoints[i]})")
    print(f"{'='*60}\n")
    
    # Initialize Pose Analyzer
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=True
        )
        print("[\u2713] Pose Analyzer initialized successfully\n")
    except Exception as e:
        print(f"[✗] Failed to initialize Pose Analyzer: {e}")
        return
    
    # Load all images
    frames = []
    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[✗] Failed to load image {i+1}: {img_path}")
            return
        frames.append(frame)
        print(f"[✓] Image {i+1} loaded: {frame.shape}")
    
    print()
    
    # Create composite frame (side-by-side zones like kiosk app)
    composite_frame = create_zoned_frame(frames, num_users)
    h, w, _ = composite_frame.shape
    print(f"[✓] Composite frame created: {w}x{h}\n")
    
    # Analyze each zone
    print("Analyzing zones...")
    zone_results = analyze_zones(composite_frame, num_users, pose_analyzer, viewpoints)
    
    # Visualize results with zoning
    visualize_multi_user_results(composite_frame, zone_results, num_users, user_names, viewpoints)


def test_single_image_multi_zone(image_path, viewpoints, user_names=None):
    """
    Test one image with multiple people in zones (realistic kiosk scenario)
    
    Args:
        image_path: Path to single image containing multiple people
        viewpoints: List of viewpoints for each zone ('front', 'left', 'right')
        user_names: Optional list of names for each user
    """
    num_users = len(viewpoints)
    
    if num_users < 2 or num_users > 3:
        print("[✗] Number of zones must be between 2 and 3")
        return
    
    if user_names is None:
        user_names = [f"User {i+1}" for i in range(num_users)]
    
    print(f"\n{'='*60}")
    print(f"Single Image Multi-Zone Test (Realistic Kiosk Mode)")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Number of Zones: {num_users}")
    for i in range(num_users):
        print(f"  Zone {i+1}: {user_names[i]} ({viewpoints[i]})")
    print(f"{'='*60}\n")
    
    # Initialize Pose Analyzer
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=True
        )
        print("[\u2713] Pose Analyzer initialized successfully\n")
    except Exception as e:
        print(f"[✗] Failed to initialize Pose Analyzer: {e}")
        return
    
    # Load the composite image
    composite_frame = cv2.imread(image_path)
    if composite_frame is None:
        print(f"[✗] Failed to load image: {image_path}")
        return
    
    h, w, _ = composite_frame.shape
    print(f"[✓] Image loaded: {w}x{h}")
    print(f"[✓] Splitting into {num_users} zones...\n")
    
    # Analyze each zone
    print("Analyzing zones...")
    zone_results = analyze_zones(composite_frame, num_users, pose_analyzer, viewpoints)
    
    # Visualize results with zoning
    visualize_multi_user_results(composite_frame, zone_results, num_users, user_names, viewpoints)


def create_zoned_frame(frames, num_users):
    """Create side-by-side composite frame with vertical separators (like kiosk app)"""
    # Find max dimensions
    max_h = max(f.shape[0] for f in frames)
    
    # Resize all frames to same height
    resized_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h != max_h:
            scale = max_h / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, max_h))
        resized_frames.append(frame)
    
    # Calculate zone width (make all zones equal width)
    total_w = sum(f.shape[1] for f in resized_frames)
    zone_w = total_w // num_users
    
    # Resize frames to equal zone widths
    equal_frames = []
    for frame in resized_frames:
        frame = cv2.resize(frame, (zone_w, max_h))
        equal_frames.append(frame)
    
    # Concatenate horizontally
    composite = np.hstack(equal_frames)
    
    # Draw vertical separators (like kiosk app)
    for i in range(1, num_users):
        x = i * zone_w
        cv2.line(composite, (x, 0), (x, max_h), (255, 255, 255), 3)
    
    return composite


def analyze_zones(composite_frame, num_users, pose_analyzer, viewpoints):
    """Analyze each user zone separately (like kiosk app's analyze_zones method)"""
    h, w, _ = composite_frame.shape
    col_w = w // num_users
    zone_results = {}
    
    for i in range(num_users):
        print(f"\n[Zone {i+1}] Analyzing...")
        
        # Extract zone
        x_start = i * col_w
        x_end = (i + 1) * col_w
        zone_frame = composite_frame[:, x_start:x_end].copy()
        zone_h, zone_w_actual, _ = zone_frame.shape
        
        # Debug: Save individual zone frames
        debug_zone_path = f"debug_zone_{i+1}_{viewpoints[i]}.jpg"
        cv2.imwrite(debug_zone_path, zone_frame)
        print(f"[Zone {i+1}] Frame size: {zone_w_actual}x{zone_h}")
        print(f"[Zone {i+1}] Saved debug frame: {debug_zone_path}")
        
        # Set viewpoint for this zone
        viewpoint = viewpoints[i]
        if pose_analyzer.gcn_engine:
            pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            print(f"[Zone {i+1}] Viewpoint set to: {viewpoint}")
        
        # Clear ALL detection state to prevent cross-zone contamination
        # 1. Stick cache: All zones use person_id=0 in snapshot mode
        pose_analyzer._cached_stick_results.clear()
        # 2. Stick smoothing buffer: Averages coordinates across zones = garbage
        pose_analyzer.stick_buffer.clear()
        # 3. MediaPipe temporal state: Reset to treat each zone as independent image
        pose_analyzer.pose = pose_analyzer.mp_pose.Pose(
            static_image_mode=True,   # Independent image mode (no temporal tracking)
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=False     # No smoothing across zones
        )
        print(f"[Zone {i+1}] Cleared all detection state (cache, buffer, MediaPipe)")
        
        # Analyze zone
        try:
            print(f"[Zone {i+1}] Running pose detection...")
            results = pose_analyzer.process_frame(zone_frame, skip_ml_inference=False, mode='snapshot')
            print(f"[Zone {i+1}] Results type: {type(results)}, Length: {len(results) if results else 0}")
            
            if results and len(results) > 0:
                person_data = results[0]
                predicted_class = person_data.get('predicted_class', 'N/A')
                confidence = person_data.get('confidence', 0.0)
                landmarks = person_data.get('landmarks_absolute')
                stick_endpoints = person_data.get('stick_endpoints')
                
                print(f"[Zone {i+1}] ========== DEBUG START ==========")
                print(f"[Zone {i+1}] person_data keys: {list(person_data.keys())}")
                print(f"[Zone {i+1}] stick_endpoints value: {stick_endpoints}")
                print(f"[Zone {i+1}] stick_endpoints type: {type(stick_endpoints)}")
                print(f"[Zone {i+1}] stick_endpoints is None: {stick_endpoints is None}")
                print(f"[Zone {i+1}] bool(stick_endpoints): {bool(stick_endpoints)}")
                print(f"[Zone {i+1}] ========== DEBUG END ==========")
                
                # Adjust coordinates for composite frame offset
                if landmarks:
                    adjusted_landmarks = [(x + x_start, y, z) for x, y, z in landmarks]
                else:
                    adjusted_landmarks = None
                
                if stick_endpoints:
                    grip_pt, tip_pt = stick_endpoints
                    print(f"[Zone {i+1}] Original stick: grip={grip_pt}, tip={tip_pt}")
                    print(f"[Zone {i+1}] Zone x_offset: {x_start}")
                    adjusted_stick = (
                        (grip_pt[0] + x_start, grip_pt[1]),
                        (tip_pt[0] + x_start, tip_pt[1])
                    )
                    print(f"[Zone {i+1}] Adjusted stick: grip={adjusted_stick[0]}, tip={adjusted_stick[1]}")
                else:
                    adjusted_stick = None
                
                # Determine quality rating
                confidence_threshold = CONFIDENCE_THRESHOLDS.get(viewpoint, 0.35)
                pose_detected = (predicted_class != 'N/A' and 
                               predicted_class.lower() != 'no technique detected' and 
                               confidence > 0)
                
                if not pose_detected:
                    rating = "NOT DETECTED"
                    rating_color = (0, 0, 255)  # Red
                elif confidence >= 0.40:
                    rating = "EXCELLENT"
                    rating_color = (46, 204, 113)  # Green
                elif confidence >= 0.30:
                    rating = "GOOD"
                    rating_color = (0, 255, 191)  # Lime
                else:
                    rating = "FAIR"
                    rating_color = (18, 153, 243)  # Orange
                
                # Convert technical name to display name
                if pose_detected:
                    display_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
                else:
                    display_name = "No Technique"
                
                zone_results[i] = {
                    'predicted_class': predicted_class,
                    'display_name': display_name,
                    'confidence': confidence,
                    'rating': rating,
                    'rating_color': rating_color,
                    'landmarks_absolute': adjusted_landmarks,
                    'stick_endpoints': adjusted_stick,
                    'stick_detected': stick_endpoints is not None,
                    'viewpoint': viewpoint
                }
                
                print(f"[Zone {i+1}] Detected: {display_name}")
                print(f"[Zone {i+1}] Confidence: {confidence:.2%}")
                print(f"[Zone {i+1}] Rating: {rating}")
                print(f"[Zone {i+1}] Stick: {'✓ Yes' if stick_endpoints else '✗ No'}")
                
            else:
                print(f"[Zone {i+1}] ⚠️  No person detected in zone")
                print(f"[Zone {i+1}] Check debug frame: debug_zone_{i+1}_{viewpoint}.jpg")
                zone_results[i] = {
                    'predicted_class': 'N/A',
                    'display_name': 'No Detection',
                    'confidence': 0.0,
                    'rating': 'NOT DETECTED',
                    'rating_color': (0, 0, 255),
                    'landmarks_absolute': None,
                    'stick_endpoints': None,
                    'stick_detected': False,
                    'viewpoint': viewpoint
                }
                
        except Exception as e:
            print(f"[Zone {i+1}] Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            zone_results[i] = {
                'predicted_class': 'ERROR',
                'display_name': 'Error',
                'confidence': 0.0,
                'rating': 'ERROR',
                'rating_color': (0, 0, 255),
                'landmarks_absolute': None,
                'stick_endpoints': None,
                'stick_detected': False,
                'viewpoint': viewpoint
            }
    
    return zone_results


def visualize_multi_user_results(composite_frame, zone_results, num_users, user_names, viewpoints):
    """Visualize multi-user results with zoning (like kiosk app feedback screen)"""
    vis_frame = composite_frame.copy()
    h, w, _ = vis_frame.shape
    col_w = w // num_users
    
    # Draw skeletons for each zone
    for i in range(num_users):
        zone_data = zone_results.get(i, {})
        landmarks = zone_data.get('landmarks_absolute')
        stick_endpoints = zone_data.get('stick_endpoints')
        rating_color = zone_data.get('rating_color', (128, 128, 128))
        
        # Draw skeleton
        if landmarks and len(landmarks) >= 33:
            draw_skeleton(vis_frame, landmarks, rating_color)
        
        # Draw stick
        if stick_endpoints:
            grip_pt, tip_pt = stick_endpoints
            print(f"[VIS] Zone {i+1} drawing stick: grip={grip_pt}, tip={tip_pt}")
            cv2.line(vis_frame, grip_pt, tip_pt, (255, 255, 0), 4)
            cv2.circle(vis_frame, grip_pt, 8, (0, 0, 255), -1)
            cv2.circle(vis_frame, grip_pt, 10, (255, 255, 255), 2)
            cv2.circle(vis_frame, tip_pt, 8, (255, 0, 0), -1)
            cv2.circle(vis_frame, tip_pt, 10, (255, 255, 255), 2)
        else:
            print(f"[VIS] Zone {i+1} NO stick endpoints to draw")
    
    # Draw vertical separators (redraw after skeletons)
    for i in range(1, num_users):
        x = i * col_w
        cv2.line(vis_frame, (x, 0), (x, h), (255, 255, 255), 3)
    
    # Add feedback overlay for each zone (like kiosk app)
    overlay = vis_frame.copy()
    panel_height = 200
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
    
    # Add text for each zone
    for i in range(num_users):
        zone_data = zone_results.get(i, {})
        display_name = zone_data.get('display_name', 'Unknown')
        confidence = zone_data.get('confidence', 0.0)
        rating = zone_data.get('rating', 'N/A')
        rating_color = zone_data.get('rating_color', (128, 128, 128))
        stick_detected = zone_data.get('stick_detected', False)
        
        # Calculate zone center
        zone_center_x = (i * col_w) + (col_w // 2)
        y_offset = h - panel_height + 30
        
        # User name at top
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(user_names[i], font, font_scale, thickness)
        text_x = zone_center_x - (text_w // 2)
        cv2.putText(vis_frame, user_names[i], (text_x, y_offset), font, font_scale, 
                   (200, 200, 200), thickness, cv2.LINE_AA)
        
        # Pose name
        y_offset += 40
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(display_name, font, font_scale, thickness)
        text_x = zone_center_x - (text_w // 2)
        cv2.putText(vis_frame, display_name, (text_x, y_offset), font, font_scale, 
                   (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Rating
        y_offset += 45
        font_scale = 1.2
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(rating, font, font_scale, thickness)
        text_x = zone_center_x - (text_w // 2)
        cv2.putText(vis_frame, rating, (text_x, y_offset), font, font_scale, 
                   rating_color, thickness, cv2.LINE_AA)
        
        # Percentage
        y_offset += 40
        percentage = f"{int(confidence * 100)}%"
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(percentage, font, font_scale, thickness)
        text_x = zone_center_x - (text_w // 2)
        cv2.putText(vis_frame, percentage, (text_x, y_offset), font, font_scale, 
                   (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Stick indicator
        y_offset += 30
        stick_text = "[OK] Stick" if stick_detected else "[X] No Stick"
        stick_color_bgr = (39, 174, 96) if stick_detected else (149, 165, 166)
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(stick_text, font, font_scale, thickness)
        text_x = zone_center_x - (text_w // 2)
        cv2.putText(vis_frame, stick_text, (text_x, y_offset), font, font_scale, 
                   stick_color_bgr, thickness, cv2.LINE_AA)
    
    # Save result
    timestamp = int(np.random.rand() * 10000)
    output_path = f"test_multi_user_result_{timestamp}.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"\n[✓] Multi-user visualization saved to: {output_path}")
    
    # Display
    try:
        screen_height = 900
        img_height, img_width = vis_frame.shape[:2]
        aspect_ratio = img_width / img_height
        
        new_height = min(screen_height, img_height)
        new_width = int(new_height * aspect_ratio)
        
        display_frame = cv2.resize(vis_frame, (new_width, new_height))
        
        window_name = "Multi-User Recognition Test (Zoning Mode)"
        cv2.imshow(window_name, display_frame)
        print(f"\nDisplay size: {new_width}x{new_height} (original: {img_width}x{img_height})")
        print("\n" + "="*60)
        print("Press any key to close visualization...")
        print("="*60)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[!] Could not display image: {e}")


def draw_skeleton(frame, landmarks, color):
    """Draw skeleton connections and keypoints (like kiosk app)"""
    # MediaPipe connections (33 keypoints)
    connections = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 12),            # Shoulders
        (11, 23), (12, 24),  # Torso
        (23, 24),            # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_pt = landmarks[start_idx][:2]
            end_pt = landmarks[end_idx][:2]
            
            if start_pt[0] > 1 and start_pt[1] > 1 and end_pt[0] > 1 and end_pt[1] > 1:
                cv2.line(frame, tuple(map(int, start_pt)), tuple(map(int, end_pt)), color, 3)
    
    # Draw keypoints
    keypoint_fill = color
    keypoint_border = tuple(max(0, c - 50) for c in color)
    
    for lm in landmarks:
        x, y = int(lm[0]), int(lm[1])
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 6, keypoint_fill, -1)
            cv2.circle(frame, (x, y), 7, keypoint_border, 2)

def test_image_classification(image_path, viewpoint='front'):
    """
    Test GCN classification on a static image using pure recognition mode
    
    Args:
        image_path: Path to the test image
        viewpoint: 'front', 'left', or 'right'
    """
    print(f"\n{'='*60}")
    print(f"Image Classification Test - Pure Recognition Mode")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Viewpoint: {viewpoint}")
    print(f"{'='*60}\n")
    
    # Initialize Pose Analyzer
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=True
        )
        print("[✓] Pose Analyzer initialized successfully\n")
    except Exception as e:
        print(f"[✗] Failed to initialize Pose Analyzer: {e}")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[✗] Failed to load image: {image_path}")
        return
    
    print(f"[✓] Image loaded: {frame.shape}\n")
    
    # Set viewpoint
    if pose_analyzer.gcn_engine:
        pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
        print(f"[✓] Viewpoint set to: {viewpoint}\n")
    
    # NEW: Models are mirror-invariant, no flipping needed for static images
    # Training images and test images are in the same orientation
    
    # Run pose analysis
    print("Running pose analysis...")
    try:
        results = pose_analyzer.process_frame(frame, skip_ml_inference=False, mode='snapshot')
        
        if results and len(results) > 0:
            person_data = results[0]
            predicted_class = person_data.get('predicted_class', 'N/A')
            confidence = person_data.get('confidence', 0.0)
            landmarks = person_data.get('landmarks_absolute')
            stick_endpoints = person_data.get('stick_endpoints')
            
            # Pure recognition mode - confidence-based evaluation
            confidence_threshold = CONFIDENCE_THRESHOLDS.get(viewpoint, 0.35)
            
            pose_detected = (predicted_class != 'N/A' and 
                           predicted_class.lower() != 'no technique detected' and 
                           confidence > 0)
            high_confidence = (confidence >= 0.40)
            good_confidence = (confidence >= 0.30)
            
            # Determine quality rating
            if not pose_detected:
                rating = "NOT DETECTED"
                rating_color = (0, 0, 255)  # Red
            elif high_confidence:
                rating = "EXCELLENT"
                rating_color = (46, 204, 113)  # Green (BGR: #2ecc71)
            elif good_confidence:
                rating = "GOOD"
                rating_color = (0, 255, 191)  # Lime (BGR: #bfff00)
            else:
                rating = "FAIR"
                rating_color = (18, 153, 243)  # Orange (BGR: #f39c12)
            
            # Convert technical name to display name
            if pose_detected:
                display_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
            else:
                display_name = "No Technique"
            
            print(f"\n{'='*60}")
            print(f"RECOGNITION RESULTS")
            print(f"{'='*60}")
            print(f"Detected Pose: {display_name}")
            print(f"Technical Name: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Quality Rating: {rating}")
            print(f"Threshold: {confidence_threshold:.0%}")
            print(f"Landmarks Detected: {len(landmarks) if landmarks else 0}")
            print(f"Stick Detected: {'✓ Yes' if stick_endpoints else '✗ No'}")
            print(f"{'='*60}\n")
            
            # Visualize results with new feedback style
            visualize_results(frame, person_data, display_name, confidence, rating, rating_color)
            
        else:
            print("[✗] No person detected in image")
            cv2.imshow("Test Result - No Detection", frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"[✗] Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def visualize_results(frame, person_data, display_name, confidence, rating, rating_color):
    """Create visualization of results using same style as main app"""
    
    # Get landmarks
    landmarks = person_data.get('landmarks_absolute')
    stick_endpoints = person_data.get('stick_endpoints')
    
    # Create visualization on original frame
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Determine skeleton color based on rating
    if rating == "EXCELLENT":
        skeleton_color = (46, 204, 113)  # Green (BGR: #2ecc71)
        keypoint_fill = (46, 204, 113)
        keypoint_border = (0, 200, 0)
    elif rating == "GOOD":
        skeleton_color = (0, 255, 191)  # Lime (BGR: #bfff00)
        keypoint_fill = (0, 255, 191)
        keypoint_border = (0, 200, 150)
    elif rating == "FAIR":
        skeleton_color = (18, 153, 243)  # Orange (BGR: #f39c12)
        keypoint_fill = (18, 153, 243)
        keypoint_border = (10, 130, 200)
    else:  # NOT DETECTED
        skeleton_color = (0, 0, 255)  # Red
        keypoint_fill = (0, 0, 255)
        keypoint_border = (0, 0, 180)
    
    # Draw skeleton
    if landmarks and len(landmarks) >= 33:
        # MediaPipe connections (33 keypoints)
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),            # Shoulders
            (11, 23), (12, 24),  # Torso
            (23, 24),            # Hips
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_pt = landmarks[start_idx][:2]
                end_pt = landmarks[end_idx][:2]
                
                # Skip if invalid coordinates
                if start_pt[0] > 1 and start_pt[1] > 1 and end_pt[0] > 1 and end_pt[1] > 1:
                    cv2.line(vis_frame, tuple(map(int, start_pt)), tuple(map(int, end_pt)), 
                            skeleton_color, 3)
        
        # Draw keypoints
        for lm in landmarks:
            x, y = int(lm[0]), int(lm[1])
            if x > 1 and y > 1:  # Skip invalid points
                cv2.circle(vis_frame, (x, y), 6, keypoint_fill, -1)
                cv2.circle(vis_frame, (x, y), 7, keypoint_border, 2)
    
    # Draw stick if detected
    if stick_endpoints:
        grip_pt, tip_pt = stick_endpoints
        # Draw stick line (bright cyan/yellow)
        cv2.line(vis_frame, grip_pt, tip_pt, (255, 255, 0), 4)
        # Draw grip point (red circle)
        cv2.circle(vis_frame, grip_pt, 8, (0, 0, 255), -1)
        cv2.circle(vis_frame, grip_pt, 10, (255, 255, 255), 2)
        # Draw tip point (blue circle)
        cv2.circle(vis_frame, tip_pt, 8, (255, 0, 0), -1)
        cv2.circle(vis_frame, tip_pt, 10, (255, 255, 255), 2)
    
    # Create overlay panel for feedback (same style as main app)
    overlay = vis_frame.copy()
    panel_height = 200
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
    
    # Add text overlay (centered, matching main app layout)
    center_x = w // 2
    y_offset = h - panel_height + 30
    
    # Detected pose name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(display_name, font, font_scale, thickness)
    text_x = center_x - (text_w // 2)
    cv2.putText(vis_frame, display_name, (text_x, y_offset), font, font_scale, 
                (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Quality rating
    y_offset += 50
    font_scale = 2.0
    thickness = 3
    (text_w, text_h), _ = cv2.getTextSize(rating, font, font_scale, thickness)
    text_x = center_x - (text_w // 2)
    cv2.putText(vis_frame, rating, (text_x, y_offset), font, font_scale, 
                rating_color, thickness, cv2.LINE_AA)
    
    # Percentage
    y_offset += 50
    percentage = f"{int(confidence * 100)}%"
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(percentage, font, font_scale, thickness)
    text_x = center_x - (text_w // 2)
    cv2.putText(vis_frame, percentage, (text_x, y_offset), font, font_scale, 
                (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Stick detection indicator
    y_offset += 40
    stick_text = "[OK] Stick Detected" if stick_endpoints else "[X] No Stick"
    stick_color_bgr = (39, 174, 96) if stick_endpoints else (149, 165, 166)  # Green or Gray
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(stick_text, font, font_scale, thickness)
    text_x = center_x - (text_w // 2)
    cv2.putText(vis_frame, stick_text, (text_x, y_offset), font, font_scale, 
                stick_color_bgr, thickness, cv2.LINE_AA)
    
    # Save result
    timestamp = int(np.random.rand() * 10000)
    output_path = f"test_result_{timestamp}.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"[✓] Visualization saved to: {output_path}")
    
    # Display with smart resizing (fit to screen, maintain aspect ratio)
    try:
        # Get screen dimensions (approximate - works on most systems)
        screen_height = 900  # Default, safe for most screens
        
        # Calculate resize to fit screen height
        img_height, img_width = vis_frame.shape[:2]
        aspect_ratio = img_width / img_height
        
        # New dimensions: height = screen height, width maintains aspect ratio
        new_height = min(screen_height, img_height)  # Don't upscale if smaller
        new_width = int(new_height * aspect_ratio)
        
        # Resize for display
        display_frame = cv2.resize(vis_frame, (new_width, new_height))
        
        window_name = f"Recognition Test - {rating}"
        cv2.imshow(window_name, display_frame)
        print(f"\nDisplay size: {new_width}x{new_height} (original: {img_width}x{img_height})")
        print("\n" + "="*60)
        print("Press any key to close visualization...")
        print("="*60)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[!] Could not display image: {e}")


if __name__ == "__main__":
    # Check for multi-user mode flag
    if "--multi" in sys.argv or "--zone" in sys.argv:
        # Multi-user zoning mode
        if len(sys.argv) < 4:
            print("\n" + "="*60)
            print("TuroArnis Multi-User Image Test (Zoning Mode)")
            print("="*60)
            print("\nUsage Option 1: Multiple separate images")
            print("  python test_image_app.py --multi <img1> <vp1> <img2> <vp2> [<img3> <vp3>]")
            print("\nUsage Option 2: Single composite image (realistic kiosk scenario)")
            print("  python test_image_app.py --multi <composite_img> <vp1> <vp2> [<vp3>]")
            print("\nExamples:")
            print("  # Separate images - 2 users")
            print("  python test_image_app.py --multi user1.jpg front user2.jpg left")
            print("")
            print("  # Separate images - 3 users")
            print("  python test_image_app.py --multi u1.jpg front u2.jpg left u3.jpg right")
            print("")
            print("  # Single composite image - 2 zones")
            print("  python test_image_app.py --multi composite.jpg front left")
            print("")
            print("  # Single composite image - 3 zones")
            print("  python test_image_app.py --multi composite.jpg front left right")
            print("\nViewpoints:")
            print("  front - User faces camera directly")
            print("  left  - User's left side faces camera")
            print("  right - User's right side faces camera")
            print("\nNote: Single image will be split into equal zones (like kiosk app)")
            print("="*60 + "\n")
            sys.exit(1)
        
        # Parse multi-user arguments
        args = sys.argv[2:]  # Skip script name and --multi flag
        
        # Check if this is single image mode (args are: image vp1 vp2 [vp3])
        # vs multiple image mode (args are: img1 vp1 img2 vp2 [img3 vp3])
        
        # Try single image mode first: check if second arg is a viewpoint
        if len(args) >= 2 and args[1] in ['front', 'left', 'right']:
            # Single image mode: <img> <vp1> <vp2> [<vp3>]
            img_path = args[0]
            viewpoints = args[1:]
            
            # Validate image path
            if not os.path.exists(img_path):
                print(f"[✗] Image not found: {img_path}")
                sys.exit(1)
            
            # Validate all viewpoints
            for vp in viewpoints:
                if vp not in ['front', 'left', 'right']:
                    print(f"[✗] Invalid viewpoint: {vp}")
                    print("Valid viewpoints: front, left, right")
                    sys.exit(1)
            
            num_zones = len(viewpoints)
            if num_zones < 2 or num_zones > 3:
                print(f"[✗] Error: Number of zones must be 2 or 3 (got {num_zones})")
                sys.exit(1)
            
            # Run single image multi-zone test
            image_paths = [img_path]
            test_multi_user_classification(image_paths, viewpoints)
            
        else:
            # Multiple image mode: <img1> <vp1> <img2> <vp2> [<img3> <vp3>]
            if len(args) % 2 != 0:
                print("[✗] Error: Each image must have a corresponding viewpoint")
                print("Usage: --multi <img1> <viewpoint1> <img2> <viewpoint2> ...")
                sys.exit(1)
            
            num_users = len(args) // 2
            if num_users < 1 or num_users > 3:
                print(f"[✗] Error: Number of users must be between 1 and 3 (got {num_users})")
                sys.exit(1)
            
            # Extract image paths and viewpoints
            image_paths = []
            viewpoints = []
            for i in range(0, len(args), 2):
                img_path = args[i]
                viewpoint = args[i + 1]
                
                # Validate image path
                if not os.path.exists(img_path):
                    print(f"[✗] Image not found: {img_path}")
                    sys.exit(1)
                
                # Validate viewpoint
                if viewpoint not in ['front', 'left', 'right']:
                    print(f"[✗] Invalid viewpoint: {viewpoint}")
                    print("Valid viewpoints: front, left, right")
                    sys.exit(1)
                
                image_paths.append(img_path)
                viewpoints.append(viewpoint)
            
            # Run multi-user test
            test_multi_user_classification(image_paths, viewpoints)
        
    else:
        # Single-user mode (original behavior)
        if len(sys.argv) < 2:
            print("\n" + "="*60)
            print("TuroArnis Image Classification Test - Pure Recognition Mode")
            print("="*60)
            print("\nUsage: python test_image_app.py <image_path> [viewpoint]")
            print("       python test_image_app.py --multi <img1> <vp1> <img2> <vp2> ...")
            print("       python test_image_app.py --multi <composite_img> <vp1> <vp2> [<vp3>]")
            print("\nSingle-User Examples:")
            print("  python test_image_app.py training_image.jpg")
            print("  python test_image_app.py training_image.jpg front")
            print("  python test_image_app.py training_image.jpg left")
            print("\nMulti-User Examples (Separate Images):")
            print("  python test_image_app.py --multi user1.jpg front user2.jpg left")
            print("  python test_image_app.py --multi u1.jpg front u2.jpg left u3.jpg right")
            print("\nMulti-Zone Examples (Single Composite Image):")
            print("  python test_image_app.py --multi composite.jpg front left")
            print("  python test_image_app.py --multi composite.jpg front left right")
            print("\nViewpoints:")
            print("  front - User faces camera directly")
            print("  left  - User's left side faces camera")
            print("  right - User's right side faces camera")
            print("\nNote: Single composite image mode splits the image into equal zones")
            print("      (most realistic for testing actual kiosk app behavior)")
            print("="*60 + "\n")
            sys.exit(1)
        
        image_path = sys.argv[1]
        viewpoint = sys.argv[2] if len(sys.argv) > 2 else 'front'
        
        # Validate viewpoint
        if viewpoint not in ['front', 'left', 'right']:
            print(f"[✗] Invalid viewpoint: {viewpoint}")
            print("Valid viewpoints: front, left, right")
            sys.exit(1)
        
        # Validate image path
        if not os.path.exists(image_path):
            print(f"[✗] Image not found: {image_path}")
            sys.exit(1)
        
        test_image_classification(image_path, viewpoint)
