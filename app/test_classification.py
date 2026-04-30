"""
Image Classification Test with GUI Feedback Overlay
Supports both Lesson Mode (with target technique) and Free Practice Mode

USAGE (single image - backward compatible):
    python test_classification.py <image_path> [target_technique] [viewpoint]

USAGE (batch mode):
    python test_classification.py --batch img1.jpg img2.jpg --target right_chest_thrust_correct --viewpoint front --outdir results/

Examples:
    # Single image lesson:
    python test_classification.py demo_images/test.jpg crown_thrust_correct front
    
    # Single image free practice:
    python test_classification.py demo_images/test.jpg front

    # Batch lesson + free practice (one composite image per input):
    python test_classification.py --batch demo_images/*.jpg --target right_chest_thrust_correct --viewpoint front
"""

import cv2
import numpy as np
import sys
import os
import glob as glob_module

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.utils.resource_path import get_resource_path


# ---------------------------------------------------------------------------
# Drawing helpers (unchanged)
# ---------------------------------------------------------------------------
def draw_rounded_rectangle(img, x, y, w, h, color, alpha=0.8, radius=10):
    """Draw a semi-transparent rounded rectangle for text background."""
    overlay = img.copy()
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.ellipse(overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ---------------------------------------------------------------------------
# Skeleton / Stick / Arrow drawing helpers
# ---------------------------------------------------------------------------
SKELETON_CONNECTIONS = [
    (11, 13), (13, 15),   # Left arm
    (12, 14), (14, 16),   # Right arm
    (11, 12),             # Shoulders
    (11, 23), (12, 24),   # Torso
    (23, 24),             # Hips
    (23, 25), (25, 27),   # Left leg
    (24, 26), (26, 28),   # Right leg
]

# Joint name → (start_idx, vertex_idx, end_idx) for red correction arrows
JOINT_ANGLE_MAP = {
    'left_elbow':  (11, 13, 15),
    'right_elbow': (12, 14, 16),
    'left_shoulder':  (23, 11, 13),
    'right_shoulder': (24, 12, 14),
    'left_knee':   (23, 25, 27),
    'right_knee':  (24, 26, 28),
    'wrist':       (12, 14, 16),  # Default to right arm for generic wrist
}

RED_BGR = (0, 0, 255)

def draw_skeleton(frame, landmarks, color=(255, 255, 0), thickness=2):
    """Draw simplified 11-connection skeleton on frame (in-place)."""
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x1, y1 = int(landmarks[start_idx][0]), int(landmarks[start_idx][1])
            x2, y2 = int(landmarks[end_idx][0]), int(landmarks[end_idx][1])
            if x1 > 1 and y1 > 1 and x2 > 1 and y2 > 1:
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    for lm in landmarks:
        x, y = int(lm[0]), int(lm[1])
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)

def draw_stick(frame, stick_endpoints, thickness=3):
    """Draw stick grip (green) and tip (red) with connecting line on frame (in-place)."""
    if not stick_endpoints:
        return
    grip, tip = stick_endpoints
    if grip and tip:
        cv2.line(frame, tuple(map(int, grip[:2])), tuple(map(int, tip[:2])), (255, 255, 0), thickness)
        cv2.circle(frame, tuple(map(int, grip[:2])), 8, (0, 255, 0), -1)
        cv2.circle(frame, tuple(map(int, grip[:2])), 9, (255, 255, 255), 2)
        cv2.circle(frame, tuple(map(int, tip[:2])), 8, (0, 0, 255), -1)
        cv2.circle(frame, tuple(map(int, tip[:2])), 9, (255, 255, 255), 2)

def draw_red_correction_arrows(frame, feedback, landmarks):
    """Draw red arrows pointing to joints that need correction (in-place)."""
    if not feedback or 'corrections' not in feedback:
        return
    for correction in feedback['corrections']:
        joint_name = correction.get('joint')
        action = correction.get('action')
        indices = JOINT_ANGLE_MAP.get(joint_name)
        if not indices:
            continue
        start_idx, vertex_idx, end_idx = indices
        if end_idx >= len(landmarks) or vertex_idx >= len(landmarks):
            continue
        p_end = landmarks[end_idx]
        p_vertex = landmarks[vertex_idx]
        start_point = (int(p_end[0]), int(p_end[1]))
        vx = p_end[0] - p_vertex[0]
        vy = p_end[1] - p_vertex[1]
        mag = np.hypot(vx, vy)
        if mag < 1e-6:
            continue
        vx, vy = vx / mag, vy / mag
        arrow_len = 40
        if action in ['extend', 'extend_grip']:
            end_point = (int(start_point[0] + vx * arrow_len), int(start_point[1] + vy * arrow_len))
        elif action in ['flex', 'retract_grip']:
            end_point = (int(start_point[0] - vx * arrow_len), int(start_point[1] - vy * arrow_len))
        else:
            continue
        cv2.arrowedLine(frame, start_point, end_point, RED_BGR, 3, tipLength=0.3)
        cv2.circle(frame, start_point, 4, RED_BGR, -1)


def get_status_color(status):
    colors = {
        'EXCELLENT!': (0, 204, 46),
        'GOOD': (0, 255, 191),
        'WRONG TECHNIQUE': (34, 126, 230),
        'NOT DETECTED': (60, 76, 231),
        'FAIR': (34, 126, 230),
    }
    return colors.get(status, (128, 128, 128))


def draw_feedback_overlay(img, predicted_class, confidence, status, feedback_messages,
                          stick_detected, target_pose=None, is_lesson_mode=False):
    h, w = img.shape[:2]
    result = img.copy()
    WHITE = (255, 255, 255)
    GREEN = (0, 204, 46)
    GRAY = (149, 165, 166)

    # 1. STATUS BAR
    status_color = get_status_color(status)
    bar_height = 80
    result = draw_rounded_rectangle(result, 20, 20, w - 40, bar_height, status_color, alpha=0.9, radius=15)
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_size = cv2.getTextSize(status, font, 1.5, 3)[0]
    status_x = (w - status_size[0]) // 2
    cv2.putText(result, status, (status_x, 75), font, 1.5, WHITE, 3)

    # 2. TECHNIQUE NAME
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

    tech_size = cv2.getTextSize(technique_text, font, 0.8, 2)[0]
    result = draw_rounded_rectangle(result, 20, 120, tech_size[0] + 30, 40, (52, 73, 94), alpha=0.8, radius=8)
    cv2.putText(result, technique_text, (35, 150), font, 0.8, WHITE, 2)

    if target_text:
        target_size = cv2.getTextSize(target_text, font, 0.7, 2)[0]
        result = draw_rounded_rectangle(result, 20, 170, target_size[0] + 30, 35, (41, 128, 185), alpha=0.8, radius=8)
        cv2.putText(result, target_text, (35, 197), font, 0.7, WHITE, 2)

    # 3. CONFIDENCE
    conf_text = f"{confidence*100:.1f}%"
    conf_size = cv2.getTextSize(conf_text, font, 1.0, 2)[0]
    result = draw_rounded_rectangle(result, w - conf_size[0] - 50, 120, conf_size[0] + 30, 45, (46, 204, 113), alpha=0.8, radius=8)
    cv2.putText(result, conf_text, (w - conf_size[0] - 35, 155), font, 1.0, WHITE, 2)
    label_text = "Confidence"
    label_size = cv2.getTextSize(label_text, font, 0.5, 1)[0]
    cv2.putText(result, label_text, (w - label_size[0] - 35, 115), font, 0.5, WHITE, 1)

    # 4. STICK INDICATOR
    stick_text = "Stick: Detected" if stick_detected else "Stick: Not Detected"
    stick_color = GREEN if stick_detected else GRAY
    stick_size = cv2.getTextSize(stick_text, font, 0.7, 2)[0]
    result = draw_rounded_rectangle(result, 20, h - 60, stick_size[0] + 30, 40, stick_color, alpha=0.8, radius=8)
    cv2.putText(result, stick_text, (35, h - 35), font, 0.7, WHITE, 2)

    # 5. FEEDBACK MESSAGES
    if feedback_messages:
        y_start = h - 80
        for i, msg in enumerate(reversed(feedback_messages[:3])):
            msg_size = cv2.getTextSize(msg, font, 0.7, 2)[0]
            msg_x = (w - msg_size[0]) // 2
            y_pos = y_start - (i * 35)
            result = draw_rounded_rectangle(result, msg_x - 15, y_pos - 25, msg_size[0] + 30, 30, (41, 128, 185), alpha=0.8, radius=6)
            cv2.putText(result, msg, (msg_x, y_pos), font, 0.7, WHITE, 2)

    # 6. MODE INDICATOR
    mode_text = "LESSON MODE" if is_lesson_mode else "FREE PRACTICE"
    mode_color = (155, 89, 182) if is_lesson_mode else (52, 73, 94)
    mode_size = cv2.getTextSize(mode_text, font, 0.6, 1)[0]
    result = draw_rounded_rectangle(result, w - mode_size[0] - 50, h - 50, mode_size[0] + 30, 30, mode_color, alpha=0.8, radius=6)
    cv2.putText(result, mode_text, (w - mode_size[0] - 35, h - 30), font, 0.6, WHITE, 1)

    return result


# ---------------------------------------------------------------------------
# Core inference helper (no display / no save)
# ---------------------------------------------------------------------------
def run_inference(frame, target_pose, viewpoint, pose_analyzer, feedback_analyzer):
    """
    Run classification on a loaded frame.
    Returns: (annotated_img, predicted_class, confidence, status, feedback_messages, stick_detected)
    """
    is_lesson_mode = target_pose is not None

    results = pose_analyzer.process_frame(frame, skip_ml_inference=False, mode='snapshot')

    if not results or len(results) == 0:
        return (
            draw_feedback_overlay(
                frame,
                predicted_class="No Technique Detected",
                confidence=0.0,
                status="NOT DETECTED",
                feedback_messages=["No person detected in frame"],
                stick_detected=False,
                target_pose=target_pose,
                is_lesson_mode=is_lesson_mode
            ),
            "No Technique Detected", 0.0, "NOT DETECTED", ["No person detected in frame"], False
        )

    person_data = results[0]
    predicted_class = person_data.get('predicted_class', 'N/A')
    confidence = person_data.get('confidence', 0.0)
    stick_detected = person_data.get('stick_detected', False)

    # ---- Draw skeleton + stick underneath (on a copy) ----
    landmarks = person_data.get('landmarks_absolute')
    stick_endpoints = person_data.get('stick_endpoints')
    vis_frame = frame.copy()
    if landmarks and len(landmarks) >= 29:
        draw_skeleton(vis_frame, landmarks, color=(255, 255, 0), thickness=2)
    if stick_endpoints:
        draw_stick(vis_frame, stick_endpoints, thickness=3)

    threshold = feedback_analyzer.get_confidence_threshold(viewpoint)
    high_confidence = (confidence >= threshold + 0.15)
    good_confidence = (confidence >= threshold)
    pose_detected = (
        predicted_class != 'N/A' and
        predicted_class.lower() not in ('no technique detected', 'neutral') and
        confidence > 0
    )

    analysis = None
    if is_lesson_mode:
        target_key = target_pose
        correct_hit = pose_detected and (predicted_class == target_key)
        wrong_hit = pose_detected and (predicted_class != target_key)

        analysis = feedback_analyzer.analyze(
            result=person_data,
            target_form=target_key,
            confidence_threshold=threshold,
            viewpoint=viewpoint,
            gcn_engine=pose_analyzer.gcn_engine if pose_analyzer else None
        )
        prioritized = feedback_analyzer.get_prioritized_messages(analysis, max_messages=10)
        feedback_messages = [msg for msg, t in prioritized if t in ('error', 'warning')]
        if not feedback_messages:
            feedback_messages = [msg for msg, t in prioritized if t == 'suggestion']

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
        if not pose_detected:
            status = "NOT DETECTED"
        elif high_confidence:
            status = "EXCELLENT!"
        elif good_confidence:
            status = "GOOD"
        else:
            status = "FAIR"

        feedback_messages = []
        if pose_detected and predicted_class:
            fp_threshold = min(threshold + 0.15, 0.98)
            analysis = feedback_analyzer.analyze(
                result=person_data,
                target_form=predicted_class,
                confidence_threshold=fp_threshold,
                viewpoint=viewpoint,
                gcn_engine=pose_analyzer.gcn_engine if pose_analyzer else None
            )
            prioritized = feedback_analyzer.get_prioritized_messages(analysis, max_messages=10)
            feedback_messages = [msg for msg, t in prioritized if t in ('error', 'warning')]
            if not feedback_messages:
                feedback_messages = [msg for msg, t in prioritized if t == 'suggestion']

    # ---- Draw red correction arrows on wrong joints ----
    if analysis and landmarks and len(landmarks) >= 29:
        draw_red_correction_arrows(vis_frame, analysis, landmarks)

    annotated = draw_feedback_overlay(
        vis_frame,
        predicted_class=predicted_class,
        confidence=confidence,
        status=status,
        feedback_messages=feedback_messages,
        stick_detected=stick_detected,
        target_pose=target_pose,
        is_lesson_mode=is_lesson_mode
    )
    return annotated, predicted_class, confidence, status, feedback_messages, stick_detected


# ---------------------------------------------------------------------------
# Single-image test (backward compatible)
# ---------------------------------------------------------------------------
def test_classification(image_path, target_pose=None, viewpoint='front'):
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

    if not os.path.exists(image_path):
        print(f"[✗] Image not found: {image_path}")
        print(f"Current directory: {os.getcwd()}")
        demo_dir = "demo_images"
        if os.path.exists(demo_dir):
            print("Available demo images:")
            for f in os.listdir(demo_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    print(f"  - {demo_dir}/{f}")
        return

    print("[1] Initializing Analyzers...")
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(detection_interval=3, stick_model_path=stick_model_path, debug_stick=False)
        feedback_analyzer = FeedbackAnalyzer()
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

    print(f"\n[2] Loading image...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[✗] Failed to load image: {image_path}")
        return
    print(f"[✓] Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels")

    print(f"\n[3] Running classification...")
    print(f"{'-'*70}")
    try:
        annotated, predicted_class, confidence, status, feedback_messages, stick_detected = run_inference(
            frame, target_pose, viewpoint, pose_analyzer, feedback_analyzer
        )

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

        output_path = f"test_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated)
        print(f"\n[✓] Result saved: {output_path}")

        cv2.imshow("Classification Result", annotated)
        print("\nDisplaying result. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[✗] Classification failed: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Batch test: lesson + free practice → one composite image per input
# ---------------------------------------------------------------------------
def batch_test(image_paths, target_pose, viewpoint, outdir):
    """
    Run batch classification.
    For each image, produces ONE composite image (side-by-side: lesson | free practice).
    """
    os.makedirs(outdir, exist_ok=True)
    is_lesson = target_pose is not None

    print(f"\n{'='*70}")
    print("BATCH CLASSIFICATION TEST")
    print(f"{'='*70}")
    print(f"Images: {len(image_paths)}")
    print(f"Target: {target_pose if target_pose else '(Free Practice only)'}")
    print(f"Viewpoint: {viewpoint}")
    print(f"Output dir: {outdir}")
    print(f"{'='*70}\n")

    # Initialize analyzers once
    print("[1] Initializing Analyzers (shared across batch)...")
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(detection_interval=3, stick_model_path=stick_model_path, debug_stick=False)
        feedback_analyzer = FeedbackAnalyzer()
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

    summary = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {image_path}")
        if not os.path.exists(image_path):
            print(f"  [✗] Image not found, skipping.")
            summary.append((os.path.basename(image_path), "MISSING", 0.0, "N/A", "N/A", 0.0, "SKIP"))
            continue

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  [✗] Failed to load image, skipping.")
            summary.append((os.path.basename(image_path), "LOAD_FAIL", 0.0, "N/A", "N/A", 0.0, "SKIP"))
            continue

        # ---- Lesson mode ----
        if is_lesson:
            lesson_img, l_pred, l_conf, l_status, l_fb, l_stick = run_inference(
                frame.copy(), target_pose, viewpoint, pose_analyzer, feedback_analyzer
            )
        else:
            lesson_img, l_pred, l_conf, l_status, l_fb, l_stick = None, "N/A", 0.0, "N/A", [], False

        # ---- Free practice mode ----
        free_img, f_pred, f_conf, f_status, f_fb, f_stick = run_inference(
            frame.copy(), None, viewpoint, pose_analyzer, feedback_analyzer
        )

        # ---- Compose side-by-side ----
        if is_lesson and lesson_img is not None:
            h, w = lesson_img.shape[:2]
            sep_width = 10
            separator = np.full((h, sep_width, 3), 255, dtype=np.uint8)
            composite = np.hstack([lesson_img, separator, free_img])

            # Add half labels at top
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_h = 40
            label_overlay = np.full((label_h, composite.shape[1], 3), (50, 50, 50), dtype=np.uint8)
            composite = np.vstack([label_overlay, composite])

            lesson_label = "LESSON"
            free_label = "FREE PRACTICE"
            lesson_color = (155, 89, 182)  # purple-ish BGR
            free_color = (52, 73, 94)       # dark blue BGR

            l_size = cv2.getTextSize(lesson_label, font, 0.8, 2)[0]
            f_size = cv2.getTextSize(free_label, font, 0.8, 2)[0]

            cv2.putText(composite, lesson_label, ((w - l_size[0]) // 2, 30), font, 0.8, lesson_color, 2)
            cv2.putText(composite, free_label, (w + sep_width + (w - f_size[0]) // 2, 30), font, 0.8, free_color, 2)
        else:
            # No lesson target — just free-practice result
            composite = free_img

        # ---- Save ----
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(outdir, f"batch_result_{stem}.jpg")
        cv2.imwrite(out_path, composite)
        print(f"  [✓] Saved: {out_path}")

        if not is_lesson:
            match = "N/A"
        elif l_pred == target_pose:
            match = "YES"
        else:
            match = "NO"
        summary.append((
            os.path.basename(image_path),
            l_pred, l_conf, l_status,
            f_pred, f_conf, f_status,
            match
        ))

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print(f"{'='*70}")
    if is_lesson:
        header = f"{'Image':<25} | {'Lesson Pred':<22} | {'L-Conf':>6} | {'L-Status':<12} | {'Free Pred':<22} | {'F-Conf':>6} | {'Match':<5}"
        print(header)
        print("-" * len(header))
        for row in summary:
            if len(row) == 8:
                name, l_pred, l_conf, l_status, f_pred, f_conf, f_status, match = row
                print(f"{name:<25} | {l_pred:<22} | {l_conf:>6.2f} | {l_status:<12} | {f_pred:<22} | {f_conf:>6.2f} | {match:<5}")
            else:
                name, l_pred, l_conf, l_status, f_pred, f_conf, f_status = row
                print(f"{name:<25} | {l_pred:<22} | {l_conf:>6.2f} | {l_status:<12} | {f_pred:<22} | {f_conf:>6.2f} | {'N/A':<5}")
    else:
        header = f"{'Image':<25} | {'Free Pred':<22} | {'F-Conf':>6} | {'F-Status':<12}"
        print(header)
        print("-" * len(header))
        for row in summary:
            if len(row) >= 6:
                name = row[0]
                f_pred = row[4] if len(row) > 4 else row[1]
                f_conf = row[5] if len(row) > 5 else row[2]
                f_status = row[6] if len(row) > 6 else row[3]
                print(f"{name:<25} | {f_pred:<22} | {f_conf:>6.2f} | {f_status:<12}")
    print(f"{'='*70}\n")
    print(f"Done. {len(image_paths)} image(s) processed. Results in: {outdir}")


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def parse_args(argv):
    """
    Manual flag parser. Supports both old positional args and new --batch mode.
    Returns: (mode, image_paths, target_pose, viewpoint, outdir)
    mode: 'single' | 'batch'
    """
    valid_viewpoints = {'front', 'left', 'right'}
    valid_targets = {
        'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
        'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
        'right_chest_thrust_correct', 'right_elbow_block_correct',
        'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
        'solar_plexus_thrust_correct', 'neutral'
    }

    # Detect --batch mode
    if '--batch' in argv:
        batch_idx = argv.index('--batch')
        image_paths = []
        i = batch_idx + 1
        while i < len(argv) and not argv[i].startswith('--'):
            # Expand globs manually (shell may not expand on Windows)
            expanded = glob_module.glob(argv[i])
            if expanded:
                image_paths.extend(expanded)
            else:
                image_paths.append(argv[i])
            i += 1

        target_pose = None
        viewpoint = 'front'
        outdir = 'batch_results'

        if '--target' in argv:
            tidx = argv.index('--target')
            if tidx + 1 < len(argv) and not argv[tidx + 1].startswith('--'):
                target_pose = argv[tidx + 1]
        if '--viewpoint' in argv:
            vidx = argv.index('--viewpoint')
            if vidx + 1 < len(argv) and not argv[vidx + 1].startswith('--'):
                viewpoint = argv[vidx + 1].lower()
        if '--outdir' in argv:
            oidx = argv.index('--outdir')
            if oidx + 1 < len(argv) and not argv[oidx + 1].startswith('--'):
                outdir = argv[oidx + 1]

        if viewpoint not in valid_viewpoints:
            print(f"[⚠] Unknown viewpoint '{viewpoint}', using 'front'")
            viewpoint = 'front'

        return 'batch', image_paths, target_pose, viewpoint, outdir

    # ---- Single-image backward-compatible mode ----
    if len(argv) < 2:
        return 'help', [], None, 'front', 'batch_results'

    image_path = argv[1]
    target_pose = None
    viewpoint = 'front'

    if len(argv) == 2:
        pass
    elif len(argv) == 3:
        arg2 = argv[2].lower()
        if arg2 in valid_viewpoints:
            viewpoint = arg2
        elif argv[2] in valid_targets or any(t in argv[2] for t in ['_correct', '_thrust', '_block', '_stance']):
            target_pose = argv[2]
        else:
            target_pose = argv[2]
    else:
        arg2 = argv[2].lower()
        arg3 = argv[3].lower() if len(argv) > 3 else None
        if arg2 in valid_viewpoints:
            viewpoint = arg2
            if len(argv) > 3:
                target_pose = argv[3]
        else:
            target_pose = argv[2]
            if arg3 and arg3 in valid_viewpoints:
                viewpoint = arg3
            elif len(argv) > 3:
                viewpoint = argv[3]

    if viewpoint not in valid_viewpoints:
        print(f"[⚠] Unknown viewpoint '{viewpoint}', using 'front'")
        viewpoint = 'front'

    return 'single', [image_path], target_pose, viewpoint, 'batch_results'


def print_help():
    print("Image Classification Test with GUI Feedback Overlay")
    print()
    print("SINGLE IMAGE (backward compatible):")
    print("  python test_classification.py <image> [target] [viewpoint]")
    print()
    print("BATCH MODE:")
    print("  python test_classification.py --batch img1.jpg img2.jpg ...")
    print("                                [--target <technique>] [--viewpoint front|left|right]")
    print("                                [--outdir <dir>]")
    print()
    print("Examples:")
    print("  # Lesson + Free Practice batch:")
    print("  python test_classification.py --batch demo_images/*.jpg --target right_chest_thrust_correct --viewpoint front")
    print()
    print("  # Free Practice only batch:")
    print("  python test_classification.py --batch demo_images/*.jpg --viewpoint front")
    print()
    demo_dir = "demo_images"
    if os.path.exists(demo_dir):
        print("Available demo images:")
        for f in sorted(os.listdir(demo_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {demo_dir}/{f}")
    print()
    print("Valid techniques for --target:")
    for t in sorted(valid_targets_global):
        print(f"  - {t}")


valid_targets_global = {
    'crown_thrust_correct', 'left_chest_thrust_correct', 'left_elbow_block_correct',
    'left_eye_thrust_correct', 'left_knee_block_correct', 'left_temple_block_correct',
    'right_chest_thrust_correct', 'right_elbow_block_correct',
    'right_eye_thrust_correct', 'right_knee_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct', 'neutral'
}


def main():
    mode, image_paths, target_pose, viewpoint, outdir = parse_args(sys.argv)

    if mode == 'help':
        print_help()
        return

    if mode == 'batch':
        if not image_paths:
            print("[✗] No images provided for batch mode.")
            print_help()
            return
        batch_test(image_paths, target_pose, viewpoint, outdir)
    else:
        test_classification(image_paths[0], target_pose, viewpoint)


if __name__ == "__main__":
    main()
