"""
Image Classification Test with GUI Feedback Overlay
Supports both Lesson Mode (with target technique) and Free Practice Mode

Usage:
    python test_classification.py <image_path> [target_technique] [viewpoint]

Examples:
    # Lesson Mode (with target technique):
    python test_classification.py demo_images/thrust_test.jpg crown_thrust_correct front
    
    # Free Practice Mode (no target):
    python test_classification.py demo_images/thrust_test.jpg front
    python test_classification.py demo_images/block_test.jpg

Features:
    - Visual feedback overlay on image
    - Color-coded status (EXCELLENT/GOOD/WRONG TECHNIQUE/NOT DETECTED)
    - Corrective feedback messages (Lesson Mode)
    - Saves annotated result to file
    - Displays result in window
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
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.utils.resource_path import get_resource_path


def draw_rounded_rectangle(img, x, y, w, h, color, alpha=0.8, radius=10):
    """Draw a semi-transparent rounded rectangle for text background."""
    overlay = img.copy()
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Draw main rectangle
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    
    # Draw corners
    cv2.ellipse(overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, -1)
    
    # Blend with original
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def get_status_color(status):
    """Get BGR color tuple for status text/background."""
    colors = {
        'EXCELLENT!': (0, 204, 46),      # #2ecc71 - green
        'GOOD': (0, 255, 191),            # #bfff00 - yellow-green
        'WRONG TECHNIQUE': (34, 126, 230), # #e67e22 - orange
        'NOT DETECTED': (60, 76, 231),    # #e74c3c - red
        'FAIR': (34, 126, 230),           # #e67e22 - orange
    }
    return colors.get(status, (128, 128, 128))


def draw_feedback_overlay(img, predicted_class, confidence, status, feedback_messages, 
                          stick_detected, target_pose=None, is_lesson_mode=False):
    """
    Draw GUI feedback overlay on image.
    
    Args:
        img: OpenCV image (BGR)
        predicted_class: Detected technique name
        confidence: Confidence score (0.0-1.0)
        status: Status text (EXCELLENT!, GOOD, WRONG TECHNIQUE, NOT DETECTED, FAIR)
        feedback_messages: List of feedback messages
        stick_detected: Boolean
        target_pose: Target technique (for lesson mode display)
        is_lesson_mode: Whether in lesson mode
    
    Returns:
        Annotated image
    """
    h, w = img.shape[:2]
    result = img.copy()
    
    # Colors (BGR format for OpenCV)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 204, 46)
    GRAY = (149, 165, 166)
    
    # 1. STATUS BAR - Large centered status at top
    status_color = get_status_color(status)
    bar_height = 80
    result = draw_rounded_rectangle(result, 20, 20, w - 40, bar_height, status_color, alpha=0.9, radius=15)
    
    # Status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_size = cv2.getTextSize(status, font, 1.5, 3)[0]
    status_x = (w - status_size[0]) // 2
    cv2.putText(result, status, (status_x, 75), font, 1.5, WHITE, 3)
    
    # 2. TECHNIQUE NAME - Top left
    if predicted_class.lower() == 'neutral':
        display_name = "Pose not recognized"
    else:
        display_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
    if is_lesson_mode and target_pose:
        target_display = target_pose.replace('_correct', '').replace('_', ' ').title()
        technique_text = f"Detected: {display_name}"
        target_text = f"Target: {target_display}"
    else:
        technique_text = f"Technique: {display_name}"
        target_text = None
    
    # Background for technique name
    tech_size = cv2.getTextSize(technique_text, font, 0.8, 2)[0]
    result = draw_rounded_rectangle(result, 20, 120, tech_size[0] + 30, 40, (52, 73, 94), alpha=0.8, radius=8)
    cv2.putText(result, technique_text, (35, 150), font, 0.8, WHITE, 2)
    
    # Target text (lesson mode only)
    if target_text:
        target_size = cv2.getTextSize(target_text, font, 0.7, 2)[0]
        result = draw_rounded_rectangle(result, 20, 170, target_size[0] + 30, 35, (41, 128, 185), alpha=0.8, radius=8)
        cv2.putText(result, target_text, (35, 197), font, 0.7, WHITE, 2)
    
    # 3. CONFIDENCE - Top right
    conf_text = f"{confidence*100:.1f}%"
    conf_size = cv2.getTextSize(conf_text, font, 1.0, 2)[0]
    result = draw_rounded_rectangle(result, w - conf_size[0] - 50, 120, conf_size[0] + 30, 45, (46, 204, 113), alpha=0.8, radius=8)
    cv2.putText(result, conf_text, (w - conf_size[0] - 35, 155), font, 1.0, WHITE, 2)
    
    # Label "Confidence"
    label_text = "Confidence"
    label_size = cv2.getTextSize(label_text, font, 0.5, 1)[0]
    cv2.putText(result, label_text, (w - label_size[0] - 35, 115), font, 0.5, WHITE, 1)
    
    # 4. STICK INDICATOR - Bottom left
    stick_text = "Stick: Detected" if stick_detected else "Stick: Not Detected"
    stick_color = GREEN if stick_detected else GRAY
    stick_size = cv2.getTextSize(stick_text, font, 0.7, 2)[0]
    result = draw_rounded_rectangle(result, 20, h - 60, stick_size[0] + 30, 40, stick_color, alpha=0.8, radius=8)
    cv2.putText(result, stick_text, (35, h - 35), font, 0.7, WHITE, 2)
    
    # 5. FEEDBACK MESSAGES - Bottom center (growing upward)
    if feedback_messages:
        y_start = h - 80
        for i, msg in enumerate(reversed(feedback_messages[:3])):  # Max 3 messages
            msg_size = cv2.getTextSize(msg, font, 0.7, 2)[0]
            msg_x = (w - msg_size[0]) // 2
            y_pos = y_start - (i * 35)
            result = draw_rounded_rectangle(result, msg_x - 15, y_pos - 25, msg_size[0] + 30, 30, (41, 128, 185), alpha=0.8, radius=6)
            cv2.putText(result, msg, (msg_x, y_pos), font, 0.7, WHITE, 2)
    
    # 6. MODE INDICATOR - Bottom right
    mode_text = "LESSON MODE" if is_lesson_mode else "FREE PRACTICE"
    mode_color = (155, 89, 182) if is_lesson_mode else (52, 73, 94)  # Purple for lesson, dark blue for free
    mode_size = cv2.getTextSize(mode_text, font, 0.6, 1)[0]
    result = draw_rounded_rectangle(result, w - mode_size[0] - 50, h - 50, mode_size[0] + 30, 30, mode_color, alpha=0.8, radius=6)
    cv2.putText(result, mode_text, (w - mode_size[0] - 35, h - 30), font, 0.6, WHITE, 1)
    
    return result


def test_classification(image_path, target_pose=None, viewpoint='front'):
    """
    Test classification on a single image with GUI feedback overlay.
    
    Args:
        image_path: Path to image file
        target_pose: Target technique for lesson mode (e.g., 'crown_thrust_correct')
                      If None, runs in Free Practice mode
        viewpoint: 'front', 'left', or 'right' (default: 'front')
    """
    is_lesson_mode = target_pose is not None
    mode_name = "LESSON MODE" if is_lesson_mode else "FREE PRACTICE"
    
    print(f"\n{'='*70}")
    print(f"{mode_name} - Classification Test")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    if is_lesson_mode:
        print(f"Target: {target_pose}")
    print(f"Viewpoint: {viewpoint}")
    print(f"{'='*70}\n")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"[✗] Image not found: {image_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available images in demo_images/:")
        demo_dir = "demo_images"
        if os.path.exists(demo_dir):
            for f in os.listdir(demo_dir):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    print(f"  - {demo_dir}/{f}")
        return
    
    # Initialize Pose Analyzer and Feedback Analyzer
    print("[1] Initializing Analyzers...")
    print("    - Loading GCN models with 3D angle calculation")
    print("    - Loading stick detection model")
    print()
    
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=False
        )
        
        # Initialize FeedbackAnalyzer for lesson mode feedback
        feedback_analyzer = FeedbackAnalyzer()
        
        # Set viewpoint
        if pose_analyzer.gcn_engine:
            pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            print(f"[✓] GCN Engine loaded - Viewpoint set to: {viewpoint}")
        else:
            print("[✗] GCN Engine not available")
            return
            
    except Exception as e:
        print(f"[✗] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load image
    print(f"\n[2] Loading image...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[✗] Failed to load image: {image_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"[✓] Image loaded: {w}x{h} pixels")
    
    # Run classification
    print(f"\n[3] Running classification...")
    print(f"{'-'*70}")
    
    try:
        # MODE: snapshot - No temporal smoothing for accurate classification
        results = pose_analyzer.process_frame(
            frame, 
            skip_ml_inference=False, 
            mode='snapshot'
        )
        
        if not results or len(results) == 0:
            print("[✗] No person detected in image")
            # Create NOT DETECTED overlay anyway
            status = "NOT DETECTED"
            result_img = draw_feedback_overlay(
                frame, 
                predicted_class="No Technique Detected",
                confidence=0.0,
                status=status,
                feedback_messages=["No person detected in frame"],
                stick_detected=False,
                target_pose=target_pose,
                is_lesson_mode=is_lesson_mode
            )
            
            # Save and display
            output_path = f"test_result_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, result_img)
            print(f"\n[✓] Result saved: {output_path}")
            
            cv2.imshow("Classification Result", result_img)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        
        # Get first detected person
        person_data = results[0]
        predicted_class = person_data.get('predicted_class', 'N/A')
        confidence = person_data.get('confidence', 0.0)
        stick_detected = person_data.get('stick_detected', False)
        
        # Determine status and generate feedback based on mode
        if is_lesson_mode:
            # === LESSON MODE LOGIC ===
            target_key = target_pose
            threshold = feedback_analyzer.get_confidence_threshold(viewpoint)
            high_confidence = (confidence >= threshold + 0.15)
            good_confidence = (confidence >= threshold)
            pose_detected = (predicted_class != 'N/A' and 
                           predicted_class.lower() not in ('no technique detected', 'neutral') and 
                           confidence > 0)
            
            correct_hit = pose_detected and (predicted_class == target_key)
            wrong_hit = pose_detected and (predicted_class != target_key)
            
            # Generate feedback using FeedbackAnalyzer
            analysis = feedback_analyzer.analyze(
                result=person_data,
                target_form=target_key,
                confidence_threshold=threshold,
                viewpoint=viewpoint,
                gcn_engine=pose_analyzer.gcn_engine if pose_analyzer else None
            )
            
            # Build feedback messages list
            feedback_messages = []
            prioritized = feedback_analyzer.get_prioritized_messages(analysis, max_messages=10)
            feedback_messages = [msg for msg, t in prioritized if t in ('error', 'warning')]
            if not feedback_messages:
                feedback_messages = [msg for msg, t in prioritized if t == 'suggestion']
            
            # Determine status
            if wrong_hit:
                status = "WRONG TECHNIQUE"
                if not feedback_messages:
                    target_display = target_key.replace('_correct', '').replace('_', ' ').title()
                    feedback_messages = [f"Adjust to {target_display} position"]
            elif correct_hit and high_confidence:
                status = "EXCELLENT!"
            elif correct_hit:
                status = "GOOD"
                if not feedback_messages:
                    feedback_messages = ["Almost there — refine your form"]
            else:
                status = "NOT DETECTED"
                if not feedback_messages:
                    feedback_messages = ["Pose not recognized, try again"]
            
        else:
            # === FREE PRACTICE MODE LOGIC ===
            threshold = feedback_analyzer.get_confidence_threshold(viewpoint)
            high_confidence = (confidence >= threshold + 0.15)
            good_confidence = (confidence >= threshold)
            pose_detected = (predicted_class != 'N/A' and 
                           predicted_class.lower() not in ('no technique detected', 'neutral') and 
                           confidence > 0)
            
            # Determine status based on confidence only
            if not pose_detected:
                status = "NOT DETECTED"
            elif high_confidence:
                status = "EXCELLENT!"
            elif good_confidence:
                status = "GOOD"
            else:
                status = "FAIR"
            
            # Generate feedback for detected pose
            feedback_messages = []
            if pose_detected and predicted_class:
                # Use a raised threshold for free practice
                fp_threshold = min(threshold + 0.15, 0.98)
                analysis = feedback_analyzer.analyze(
                    result=person_data,
                    target_form=predicted_class,  # Compare against detected pose
                    confidence_threshold=fp_threshold,
                    viewpoint=viewpoint,
                    gcn_engine=pose_analyzer.gcn_engine if pose_analyzer else None
                )
                prioritized = feedback_analyzer.get_prioritized_messages(analysis, max_messages=10)
                feedback_messages = [msg for msg, t in prioritized if t in ('error', 'warning')]
                if not feedback_messages:
                    feedback_messages = [msg for msg, t in prioritized if t == 'suggestion']
        
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION RESULT")
        print(f"{'='*70}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"Status: {status}")
        if is_lesson_mode:
            print(f"Target: {target_pose}")
        print(f"Stick Detected: {'Yes' if stick_detected else 'No'}")
        if feedback_messages:
            print(f"Feedback: {' | '.join(feedback_messages[:3])}")
        print(f"{'='*70}")
        
        # Create visual overlay
        print(f"\n[4] Generating visual feedback overlay...")
        result_img = draw_feedback_overlay(
            frame,
            predicted_class=predicted_class,
            confidence=confidence,
            status=status,
            feedback_messages=feedback_messages,
            stick_detected=stick_detected,
            target_pose=target_pose,
            is_lesson_mode=is_lesson_mode
        )
        
        # Save result
        output_path = f"test_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_img)
        print(f"[✓] Result saved: {output_path}")
        
        # Display in window
        cv2.imshow("Classification Result", result_img)
        print(f"\nDisplaying result. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[✗] Classification failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Parse arguments and run test"""
    if len(sys.argv) < 2:
        print("Image Classification Test with GUI Feedback Overlay")
        print()
        print("Usage: python test_classification.py <image_path> [target_technique] [viewpoint]")
        print()
        print("Examples:")
        print("  # Lesson Mode (with target technique):")
        print("  python test_classification.py demo_images/test.jpg crown_thrust_correct front")
        print()
        print("  # Free Practice Mode (no target):")
        print("  python test_classification.py demo_images/test.jpg front")
        print("  python test_classification.py demo_images/test.jpg")
        print()
        print("Arguments:")
        print("  image_path       - Path to image file (jpg, png, etc.)")
        print("  target_technique - Target technique for lesson mode (optional)")
        print("                     Examples: crown_thrust_correct, left_chest_thrust_correct")
        print("  viewpoint        - 'front', 'left', or 'right' (default: front)")
        print()
        
        # List available demo images
        demo_dir = "demo_images"
        if os.path.exists(demo_dir):
            print("Available demo images:")
            for f in sorted(os.listdir(demo_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"  - {demo_dir}/{f}")
        print()
        print("Available techniques for lesson mode:")
        print("  - crown_thrust_correct")
        print("  - left_chest_thrust_correct")
        print("  - right_chest_thrust_correct")
        print("  - left_eye_thrust_correct")
        print("  - right_eye_thrust_correct")
        print("  - solar_plexus_thrust_correct")
        print("  - left_elbow_block_correct")
        print("  - right_elbow_block_correct")
        print("  - etc.")
        return
    
    image_path = sys.argv[1]
    
    # Parse arguments: support both orderings for flexibility
    # Format 1: image target viewpoint
    # Format 2: image viewpoint (no target - free practice)
    
    valid_viewpoints = ['front', 'left', 'right']
    valid_targets = [
        'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
        'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
        'right_chest_thrust_correct', 'right_elbow_block_correct',
        'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
        'solar_plexus_thrust_correct', 'neutral'
    ]
    
    target_pose = None
    viewpoint = 'front'
    
    if len(sys.argv) == 2:
        # Just image provided - use defaults
        pass
    elif len(sys.argv) == 3:
        # Could be: image target OR image viewpoint
        arg2 = sys.argv[2].lower()
        if arg2 in valid_viewpoints:
            viewpoint = arg2
        elif sys.argv[2] in valid_targets or any(t in sys.argv[2] for t in ['_correct', '_thrust', '_block', '_stance']):
            target_pose = sys.argv[2]
        else:
            # Assume it's a target (lesson mode)
            target_pose = sys.argv[2]
    else:
        # 3+ arguments: assume image target viewpoint
        # Check if arg2 is a valid target or viewpoint
        arg2 = sys.argv[2].lower()
        arg3 = sys.argv[3].lower() if len(sys.argv) > 3 else None
        
        if arg2 in valid_viewpoints:
            # Format: image viewpoint target (unusual but handle it)
            viewpoint = arg2
            if len(sys.argv) > 3:
                target_pose = sys.argv[3]
        else:
            # Standard format: image target viewpoint
            target_pose = sys.argv[2]
            if arg3 and arg3 in valid_viewpoints:
                viewpoint = arg3
            elif len(sys.argv) > 3:
                # Invalid viewpoint, but use it anyway with warning
                viewpoint = sys.argv[3]
    
    # Validate viewpoint
    if viewpoint not in valid_viewpoints:
        print(f"[⚠] Unknown viewpoint '{viewpoint}', using 'front'")
        viewpoint = 'front'
    
    # Show mode
    if target_pose:
        print(f"Mode: Lesson Mode (target: {target_pose})")
    else:
        print(f"Mode: Free Practice")
    print(f"Viewpoint: {viewpoint}")
    print()
    
    test_classification(image_path, target_pose, viewpoint)


if __name__ == "__main__":
    main()
