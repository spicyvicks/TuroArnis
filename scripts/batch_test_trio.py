"""
Batch Test: All 12 Arnis Poses × 3 Viewpoints (Left, Front, Right)

Runs each composite image in trio/ through the multi-zone pipeline and
produces a summary table showing prediction accuracy per viewpoint.
Output images match the exact style of test_multi_user_result from test_image_app.py.

Zone layout per image:
  Zone 1 (Left)  = left viewpoint   — correct form
  Zone 2 (Center)= front viewpoint  — correct form
  Zone 3 (Right) = right viewpoint  — INTENTIONALLY INCORRECT form

Usage:
  python scripts/batch_test_trio.py
  python scripts/batch_test_trio.py --no-display
  python scripts/batch_test_trio.py --save-dir results_trio
"""
import cv2
import numpy as np
import sys
import os
import time
import argparse
import json
from collections import OrderedDict

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.utils.resource_path import get_resource_path

# ── Configuration ──────────────────────────────────────────────────────────────

TRIO_DIR = os.path.join(project_root, "trio")

# Filename → expected GCN class name mapping
FILENAME_TO_CLASS = {
    "crown":         "crown_thrust_correct",
    "left_chest":    "left_chest_thrust_correct",
    "left_elbow":    "left_elbow_block_correct",
    "left_eye":      "left_eye_thrust_correct",
    "left_knee":     "left_knee_block_correct",
    "left_temple":   "left_temple_block_correct",
    "right_chest":   "right_chest_thrust_correct",
    "right_elbow":   "right_elbow_block_correct",
    "right_eye":     "right_eye_thrust_correct",
    "right_knee":    "right_knee_block_correct",
    "right_temple":  "right_temple_block_correct",
    "solar_plexus":  "solar_plexus_thrust_correct",
}

# Zone ↔ viewpoint mapping  (zone index: 0=left, 1=front, 2=right)
ZONE_VIEWPOINTS = ["left", "front", "right"]
ZONE_USER_NAMES = ["User 1", "User 2", "User 3"]

# Confidence thresholds are defined per-viewpoint in app/models/gcn_model_config.json
# (single source of truth — do not duplicate here)

# ── Skeleton drawing helper (identical to test_image_app.py) ───────────────────

CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (11, 23), (12, 24),  # Torso
    (23, 24),            # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]


def draw_skeleton(frame, landmarks, color):
    """Draw skeleton connections and keypoints (identical to test_image_app.py)"""
    for start_idx, end_idx in CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_pt = landmarks[start_idx][:2]
            end_pt = landmarks[end_idx][:2]
            if start_pt[0] > 1 and start_pt[1] > 1 and end_pt[0] > 1 and end_pt[1] > 1:
                cv2.line(frame, tuple(map(int, start_pt)), tuple(map(int, end_pt)), color, 3)
    keypoint_fill = color
    keypoint_border = tuple(max(0, c - 50) for c in color)
    for lm in landmarks:
        x, y = int(lm[0]), int(lm[1])
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 6, keypoint_fill, -1)
            cv2.circle(frame, (x, y), 7, keypoint_border, 2)


def get_rating(predicted_class, confidence, viewpoint='front', pose_analyzer=None):
    """Determine quality rating matching app.py show_feedback logic.
    Reads per-viewpoint threshold from gcn_model_config.json (single source of truth)."""
    pose_detected = (predicted_class != 'N/A' and
                     predicted_class.lower() != 'no technique detected' and
                     predicted_class not in ('NO_DETECTION', 'ERROR') and
                     confidence > 0)

    if not pose_detected:
        return "NOT DETECTED", (0, 0, 255)

    # Read threshold from GCN engine config (mirrors app.py show_feedback)
    gcn_config = pose_analyzer.gcn_engine.config if (pose_analyzer and pose_analyzer.gcn_engine) else {}
    threshold = gcn_config.get('models', {}).get(viewpoint, {}).get('confidence_threshold', 0.55)

    if confidence >= threshold + 0.15:   # Excellent band
        return "EXCELLENT", (46, 204, 113)   # Green
    elif confidence >= threshold:         # Good band (minimum)
        return "GOOD", (0, 255, 191)         # Lime
    else:
        return "FAIR", (18, 153, 243)        # Orange


# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze_single_image(image_path, pose_analyzer, pose_stem=None, stick_config=None):
    """
    Analyze one trio image (3 zones: left, front, right).
    Returns dict {0: zone_data, 1: zone_data, 2: zone_data} matching
    the exact format used by test_image_app.py's analyze_zones.
    
    Args:
        image_path: Path to composite image
        pose_analyzer: PoseAnalyzer instance
        pose_stem: Filename stem (e.g., 'crown') for pose lookup
        stick_config: Optional dict with per-pose stick visualization settings
    """
    composite = cv2.imread(image_path)
    if composite is None:
        print(f"  [✗] Could not load {image_path}")
        return {}

    h, w, _ = composite.shape
    col_w = w // 3
    zone_results = {}

    for zone_idx in range(3):
        viewpoint = ZONE_VIEWPOINTS[zone_idx]
        print(f"\n[Zone {zone_idx+1}] Analyzing...")

        # Extract zone
        x_start = zone_idx * col_w
        x_end = (zone_idx + 1) * col_w
        zone_frame = composite[:, x_start:x_end].copy()
        zone_h, zone_w_actual, _ = zone_frame.shape
        print(f"[Zone {zone_idx+1}] Frame size: {zone_w_actual}x{zone_h}")

        # Set viewpoint
        if pose_analyzer.gcn_engine:
            pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            print(f"[Zone {zone_idx+1}] Viewpoint set to: {viewpoint}")

        # Clear ALL detection state to prevent cross-zone contamination
        pose_analyzer._cached_stick_results.clear()
        pose_analyzer.stick_buffer.clear()
        pose_analyzer.pose = pose_analyzer.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=False,
        )
        print(f"[Zone {zone_idx+1}] Cleared all detection state (cache, buffer, MediaPipe)")

        try:
            print(f"[Zone {zone_idx+1}] Running pose detection...")
            results = pose_analyzer.process_frame(zone_frame, skip_ml_inference=False, mode='snapshot',
                                                 target_pose=pose_stem, stick_hand_config=stick_config)

            if results and len(results) > 0:
                person_data = results[0]
                predicted_class = person_data.get('predicted_class', 'N/A')
                confidence = person_data.get('confidence', 0.0)
                landmarks = person_data.get('landmarks_absolute')
                stick_endpoints = person_data.get('stick_endpoints')

                # Adjust coordinates for composite frame offset
                if landmarks:
                    adjusted_landmarks = [(x + x_start, y, z) for x, y, z in landmarks]
                else:
                    adjusted_landmarks = None

                if stick_endpoints:
                    grip_pt, tip_pt = stick_endpoints
                    adjusted_stick = (
                        (grip_pt[0] + x_start, grip_pt[1]),
                        (tip_pt[0] + x_start, tip_pt[1])
                    )
                else:
                    adjusted_stick = None

                # Determine quality rating (same logic as app.py show_feedback)
                rating, rating_color = get_rating(predicted_class, confidence, viewpoint=viewpoint, pose_analyzer=pose_analyzer)

                # Convert technical name to display name
                pose_detected = (predicted_class != 'N/A' and
                                 predicted_class.lower() != 'no technique detected' and
                                 confidence > 0)
                if pose_detected:
                    display_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
                else:
                    display_name = "No Technique"

                zone_results[zone_idx] = {
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

                print(f"[Zone {zone_idx+1}] Detected: {display_name}")
                print(f"[Zone {zone_idx+1}] Confidence: {confidence:.2%}")
                print(f"[Zone {zone_idx+1}] Rating: {rating}")
                print(f"[Zone {zone_idx+1}] Stick: {'✓ Yes' if stick_endpoints else '✗ No'}")
            else:
                print(f"[Zone {zone_idx+1}] No person detected in zone")
                zone_results[zone_idx] = {
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
            print(f"[Zone {zone_idx+1}] Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            zone_results[zone_idx] = {
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


def visualize_and_save(composite_frame, zone_results, user_names, viewpoints, save_path, show=False, window_title=None, stick_config=None):
    """
    Visualize multi-user results — IDENTICAL layout to test_image_app.py's
    visualize_multi_user_results function.
    
    Args:
        stick_config: Optional dict mapping class names to stick visualization settings
                     {'mode': 'auto'|'force_circle'|'force_line', 'foreshortening_threshold': int}
    """
    num_users = 3
    vis_frame = composite_frame.copy()
    h, w, _ = vis_frame.shape
    col_w = w // num_users

    # Draw skeletons for each zone
    for i in range(num_users):  
        zone_data = zone_results.get(i, {})
        landmarks = zone_data.get('landmarks_absolute')
        stick_endpoints = zone_data.get('stick_endpoints')
        rating_color = zone_data.get('rating_color', (128, 128, 128))
        predicted_class = zone_data.get('predicted_class', 'N/A')
        viewpoint = viewpoints[i] if i < len(viewpoints) else 'unknown'

        # Draw skeleton
        if landmarks and len(landmarks) >= 33:
            draw_skeleton(vis_frame, landmarks, rating_color)

        # Draw stick (apply per-pose config if provided)
        if stick_endpoints:
            stick_foreshortened = zone_data.get('stick_foreshortened', False)
            skip_drawing = False
            
            # Apply stick visualization config — FRONT VIEW ONLY
            # Config was designed for front-view correct classifications (e.g. thrusts pointing at camera)
            # Left/right views always use 'auto' (foreshortening flag decides line vs circle)
            if stick_config and predicted_class in stick_config and viewpoint == 'front':
                pose_config = stick_config[predicted_class]
                
                # Check for viewpoint-specific override first
                if 'viewpoint_overrides' in pose_config and viewpoint in pose_config['viewpoint_overrides']:
                    mode = pose_config['viewpoint_overrides'][viewpoint]
                    print(f"[VIZ] Zone {i+1} ({viewpoint}): Using viewpoint override mode={mode} for {predicted_class}")
                else:
                    mode = pose_config.get('mode', 'auto')
                    print(f"[VIZ] Zone {i+1} ({viewpoint}): Using default mode={mode} for {predicted_class}")
                
                if mode == 'force_circle':
                    stick_foreshortened = True  # Always draw circle
                elif mode == 'force_line':
                    stick_foreshortened = False  # Always draw line
                elif mode == 'skip':
                    skip_drawing = True  # Don't draw anything
                # 'auto' mode uses the original stick_foreshortened value
            
            if not skip_drawing:
                grip_pt, tip_pt = stick_endpoints
                
                if stick_foreshortened:
                    # Stick pointing at camera - draw indicator circle at grip
                    cv2.circle(vis_frame, grip_pt, 20, (255, 255, 0), 3)  # Yellow circle
                    cv2.circle(vis_frame, grip_pt, 8, (0, 0, 255), -1)   # Red center
                    cv2.putText(vis_frame, "stick->camera", (grip_pt[0]-50, grip_pt[1]+35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    # Normal stick visualization
                    cv2.line(vis_frame, grip_pt, tip_pt, (255, 255, 0), 4)
                    cv2.circle(vis_frame, grip_pt, 8, (0, 0, 255), -1)
                    cv2.circle(vis_frame, grip_pt, 10, (255, 255, 255), 2)
                    cv2.circle(vis_frame, tip_pt, 8, (255, 0, 0), -1)
                    cv2.circle(vis_frame, tip_pt, 10, (255, 255, 255), 2)

    # Draw vertical separators (redraw after skeletons)
    for i in range(1, num_users):
        x = i * col_w
        cv2.line(vis_frame, (x, 0), (x, h), (255, 255, 255), 3)

    # Add feedback overlay for each zone (like kiosk app)
    overlay = vis_frame.copy()
    panel_height = 200
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)

    # Add text for each zone — IDENTICAL to test_image_app.py
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

        font = cv2.FONT_HERSHEY_SIMPLEX

        # User name at top
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
    cv2.imwrite(save_path, vis_frame)
    print(f"\n[✓] Multi-user visualization saved to: {save_path}")

    # Display (fit to screen without stretching)
    if show:
        try:
            # Leave headroom for taskbar and window chrome; keep feedback panel visible
            screen_width = 1280
            screen_height = 720

            img_height, img_width = vis_frame.shape[:2]

            # Scale to fit within screen bounds (preserves aspect ratio)
            scale = min(screen_width / img_width, screen_height / img_height)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            display_frame = cv2.resize(vis_frame, (new_width, new_height), interpolation=interp)

            win_name = window_title or "Multi-User Recognition Test (Zoning Mode)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, new_width, new_height)
            cv2.imshow(win_name, display_frame)
            print(f"\nDisplay size: {new_width}x{new_height} (original: {img_width}x{img_height}, scale: {scale:.2f}x)")
            print("\n" + "=" * 60)
            print("Press any key to continue to next pose (or 'q' to quit)...")
            print("=" * 60)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            return key
        except Exception as e:
            print(f"[!] Could not display image: {e}")
    return None


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary_table(all_results):
    """Print a formatted table for the paper."""

    # Header
    print("\n" + "=" * 120)
    print("BATCH TEST RESULTS — 12 Arnis Poses × 3 Viewpoints")
    print("Zone 1 = Left viewpoint (correct)  |  Zone 2 = Front viewpoint (correct)  |  Zone 3 = Right viewpoint (intentional error)")
    print("=" * 120)

    header = f"{'Pose':<22} | {'Left View (Zone 1)':<28} | {'Front View (Zone 2)':<28} | {'Right View (Zone 3 - Error)':<32}"
    print(header)
    print("-" * 120)

    correct_left = 0
    correct_front = 0
    total = len(all_results)
    stick_counts = {"left": 0, "front": 0, "right": 0}

    for pose_name, (expected, zones) in all_results.items():
        cols = []
        for z_idx in range(3):
            zr = zones.get(z_idx)
            if zr is None:
                cols.append("—")
                continue
            pred = zr["predicted_class"]
            conf = zr["confidence"]
            stick = "S" if zr["stick_detected"] else "-"
            match = (pred == expected)
            short = pred.replace("_correct", "").replace("_", " ").title() if pred not in ("NO_DETECTION", "ERROR", "N/A") else pred

            if zr["stick_detected"]:
                stick_counts[ZONE_VIEWPOINTS[z_idx]] += 1

            if z_idx < 2:  # left / front — expect match
                icon = "OK" if match else "X "
                if match:
                    if z_idx == 0:
                        correct_left += 1
                    else:
                        correct_front += 1
            else:           # right — intentional error
                icon = "ERR" if not match else "OK?"

            cols.append(f"{icon} {short} ({conf:.0%}) [{stick}]")

        row = f"{pose_name:<22} | {cols[0]:<28} | {cols[1]:<28} | {cols[2]:<32}"
        print(row)

    print("-" * 120)
    print(f"Left accuracy  (Zone 1): {correct_left}/{total} ({correct_left/total:.0%})")
    print(f"Front accuracy (Zone 2): {correct_front}/{total} ({correct_front/total:.0%})")
    print(f"Right (Zone 3): intentionally incorrect — mismatch expected")
    print(f"Stick detection: Left {stick_counts['left']}/{total} | Front {stick_counts['front']}/{total} | Right {stick_counts['right']}/{total}")
    print("=" * 120 + "\n")

    return correct_left, correct_front


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch test all 12 trio images")
    parser.add_argument("--no-display", action="store_true", help="Skip cv2.imshow windows")
    parser.add_argument("--save-dir", default="results_trio", help="Directory to save annotated images (default: results_trio)")
    args = parser.parse_args()

    save_dir = os.path.join(project_root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Load stick visualization configuration
    stick_config = None
    stick_config_path = os.path.join(project_root, 'scripts', 'stick_visualization_config.json')
    if os.path.exists(stick_config_path):
        try:
            with open(stick_config_path, 'r') as f:
                stick_config = json.load(f)
            print(f"\n[STICK-VIZ] Loaded stick visualization config from {stick_config_path}")
        except Exception as e:
            print(f"\n[WARN] Failed to load stick visualization config: {e}")
            print("[WARN] Using automatic foreshortening detection...")
    else:
        print(f"\n[INFO] No stick visualization config found, using automatic detection")

    # Discover images
    image_files = sorted([
        f for f in os.listdir(TRIO_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"\nFound {len(image_files)} images in {TRIO_DIR}/")
    for f in image_files:
        print(f"  - {f}")

    # Initialize PoseAnalyzer once (expensive)
    print("\nInitializing PoseAnalyzer...")
    t0 = time.time()
    stick_model_path = get_resource_path("deployment_package/weights/best.pt")
    pose_analyzer = PoseAnalyzer(
        detection_interval=3,
        stick_model_path=stick_model_path,
        debug_stick=True,
    )
    print(f"PoseAnalyzer ready in {time.time()-t0:.1f}s\n")

    # Process each image
    all_results = OrderedDict()

    for idx, fname in enumerate(image_files, 1):
        stem = os.path.splitext(fname)[0]           # e.g. "crown"
        expected_class = FILENAME_TO_CLASS.get(stem)
        if expected_class is None:
            print(f"[{idx}/{len(image_files)}] SKIP {fname} — no expected class mapping")
            continue

        image_path = os.path.join(TRIO_DIR, fname)
        display_name = expected_class.replace("_correct", "").replace("_", " ").title()

        print(f"\n{'='*60}")
        print(f"[{idx}/{len(image_files)}] {display_name}  ({fname})")
        print(f"{'='*60}")

        # Analyze all 3 zones
        zone_results = analyze_single_image(image_path, pose_analyzer, 
                                           pose_stem=stem, stick_config=stick_config)

        # Print per-zone summary
        for z_idx in range(3):
            zr = zone_results.get(z_idx)
            if zr is None:
                status = "—"
            else:
                match = zr["predicted_class"] == expected_class
                icon = "OK" if match else "X"
                status = f'{icon} {zr["predicted_class"]} ({zr["confidence"]:.0%})'
            vp = ZONE_VIEWPOINTS[z_idx]
            print(f"  Zone {z_idx+1} ({vp:>5}): {status}")

        # Visualize and save — IDENTICAL style to test_multi_user_result
        composite = cv2.imread(image_path)
        save_path = os.path.join(save_dir, f"test_multi_user_result_{stem}.jpg")
        window_title = f"Recognition Test [{idx}/{len(image_files)}]: {display_name}"

        key = visualize_and_save(
            composite, zone_results, ZONE_USER_NAMES, ZONE_VIEWPOINTS,
            save_path, show=(not args.no_display), window_title=window_title,
            stick_config=stick_config
        )

        all_results[display_name] = (expected_class, zone_results)

        # Allow quitting mid-batch
        if key == ord("q"):
            print("\n[!] User quit — stopping batch early.")
            break

    # Summary table
    correct_left, correct_front = print_summary_table(all_results)

    print(f"Annotated images saved to: {save_dir}/")
    print(f"Total images processed: {len(all_results)}")


if __name__ == "__main__":
    main()
