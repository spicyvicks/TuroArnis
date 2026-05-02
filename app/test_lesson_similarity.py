#!/usr/bin/env python3
"""
Lesson Similarity Test Script
Tests similarity-based lesson mode on a single image or a batch of images.
Outputs: similarity score, status, detected class, and actionable tips per person.

USAGE (single image — interactive):
    python test_lesson_similarity.py <image_path> <target_technique> [viewpoint]

USAGE (batch — non-interactive):
    python test_lesson_similarity.py --batch img1.jpg img2.jpg ... --target <target_technique> [--viewpoint front] [--outdir results/]

    # Or process a whole folder:
    python test_lesson_similarity.py --batch demo_images/ --target right_chest_thrust_correct --viewpoint front
"""
import argparse
import cv2
import glob as glob_module
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_mapper import generate_lesson_tips
from app.models.gcn.feature_extraction import compute_global_features_from_kpts
from app.utils.resource_path import get_resource_path


# ── Classifier-based lesson status thresholds ──
# Primary: Did the user do the correct technique?
LESSON_THRESHOLDS = [
    (0.85, "EXCELLENT!",  (0, 204, 46)),
    (0.70, "GOOD",        (0, 255, 191)),
    (0.55, "FAIR",        (34, 126, 230)),
]

# ── Mirror-class mapping for front-view symmetry diagnosis ──
OPPOSITE_CLASS = {
    'left_chest_thrust_correct':   'right_chest_thrust_correct',
    'right_chest_thrust_correct':  'left_chest_thrust_correct',
    'left_elbow_block_correct':    'right_elbow_block_correct',
    'right_elbow_block_correct':   'left_elbow_block_correct',
    'left_eye_thrust_correct':     'right_eye_thrust_correct',
    'right_eye_thrust_correct':    'left_eye_thrust_correct',
    'left_knee_block_correct':     'right_knee_block_correct',
    'right_knee_block_correct':    'left_knee_block_correct',
    'left_temple_block_correct':   'right_temple_block_correct',
    'right_temple_block_correct':  'left_temple_block_correct',
}

def get_lesson_status(predicted_class, target_pose, classifier_conf):
    """Primary lesson status: did the user do the right technique?"""
    if predicted_class == target_pose:
        for thresh, label, color in LESSON_THRESHOLDS:
            if classifier_conf >= thresh:
                return label, color
        return "KEEP TRYING", (60, 76, 231)
    else:
        return "WRONG TECHNIQUE", (0, 0, 255)  # Red


def draw_rounded_rectangle(img, x, y, w, h, color, alpha=0.8, radius=10):
    overlay = img.copy()
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.ellipse(overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_similarity_overlay(img, target_pose, form_quality, display_quality,
                            status, status_color, tips, stick_detected,
                            predicted_class, classifier_conf,
                            opposite_score, opposite_class):
    h, w = img.shape[:2]
    result = img.copy()
    WHITE = (255, 255, 255)
    GREEN = (0, 204, 46)
    GRAY = (149, 165, 166)
    WARN = (0, 0, 255)      # Red warning
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. STATUS BAR (classifier-driven)
    bar_height = 80
    result = draw_rounded_rectangle(result, 20, 20, w - 40, bar_height,
                                    status_color, alpha=0.9, radius=15)
    status_size = cv2.getTextSize(status, font, 1.5, 3)[0]
    status_x = (w - status_size[0]) // 2
    cv2.putText(result, status, (status_x, 75), font, 1.5, WHITE, 3)

    # Helper to format class names
    def fmt(name):
        if not name or name.lower() == 'n/a':
            return "N/A"
        return name.replace('_correct', '').replace('_', ' ').title()

    # 2. DETECTED LINE
    detected_text = f"Detected: {fmt(predicted_class)} @ {classifier_conf*100:.1f}%"
    det_size = cv2.getTextSize(detected_text, font, 0.75, 2)[0]
    result = draw_rounded_rectangle(result, 20, 105, det_size[0] + 30, 35,
                                    (128, 128, 128), alpha=0.8, radius=8)
    cv2.putText(result, detected_text, (35, 130), font, 0.75, WHITE, 2)

    # 3. TARGET LINE
    target_display = fmt(target_pose)
    target_text = f"Target: {target_display}"
    target_size = cv2.getTextSize(target_text, font, 0.75, 2)[0]
    result = draw_rounded_rectangle(result, 20, 148, target_size[0] + 30, 35,
                                    (52, 73, 94), alpha=0.8, radius=8)
    cv2.putText(result, target_text, (35, 173), font, 0.75, WHITE, 2)

    # 4. FORM QUALITY (template similarity — secondary diagnostic)
    fq_text = f"{form_quality:.1f}%"
    fq_size = cv2.getTextSize(fq_text, font, 1.2, 3)[0]
    result = draw_rounded_rectangle(result, w - fq_size[0] - 50, 105,
                                    fq_size[0] + 30, 78,
                                    (46, 204, 113), alpha=0.8, radius=8)
    cv2.putText(result, fq_text, (w - fq_size[0] - 35, 150), font, 1.2, WHITE, 3)
    label_text = "Form Quality"
    label_size = cv2.getTextSize(label_text, font, 0.45, 1)[0]
    cv2.putText(result, label_text,
                (w - label_size[0] - 35, 100), font, 0.45, WHITE, 1)

    # 5. OPPOSITE CLASS SIMILARITY (diagnostic)
    if opposite_class and opposite_score is not None:
        opp_text = f"{fmt(opposite_class)}: {opposite_score:.1f}%"
        opp_size = cv2.getTextSize(opp_text, font, 0.65, 2)[0]
        # Highlight in red if opposite is higher than target (symmetry confusion)
        opp_color = WARN if opposite_score > form_quality else (128, 128, 128)
        result = draw_rounded_rectangle(result, w - opp_size[0] - 50, 192,
                                        opp_size[0] + 30, 30,
                                        opp_color, alpha=0.8, radius=6)
        cv2.putText(result, opp_text, (w - opp_size[0] - 35, 215), font, 0.65, WHITE, 2)
        opp_label = "Opposite"
        opp_label_size = cv2.getTextSize(opp_label, font, 0.45, 1)[0]
        cv2.putText(result, opp_label,
                    (w - opp_label_size[0] - 35, 188), font, 0.45, WHITE, 1)

    # 6. STICK INDICATOR
    stick_text = "Stick: Detected" if stick_detected else "Stick: Not Detected"
    stick_color = GREEN if stick_detected else GRAY
    stick_size = cv2.getTextSize(stick_text, font, 0.7, 2)[0]
    result = draw_rounded_rectangle(result, 20, h - 60, stick_size[0] + 30, 40,
                                    stick_color, alpha=0.8, radius=8)
    cv2.putText(result, stick_text, (35, h - 35), font, 0.7, WHITE, 2)

    # 7. TIPS
    if tips:
        y_start = h - 80
        for i, msg in enumerate(reversed(tips[:3])):
            msg_size = cv2.getTextSize(msg, font, 0.7, 2)[0]
            msg_x = (w - msg_size[0]) // 2
            y_pos = y_start - (i * 35)
            result = draw_rounded_rectangle(result, msg_x - 15, y_pos - 25,
                                            msg_size[0] + 30, 30,
                                            (41, 128, 185), alpha=0.8, radius=6)
            cv2.putText(result, msg, (msg_x, y_pos), font, 0.7, WHITE, 2)

    return result


def _compute_person_similarity(person_data, frame, target_pose, viewpoint, gcn_engine):
    """Compute similarity + tips for a single detected person.
    Returns (annotated, result_dict) or (None, None) on failure.
    """
    h_frame, w_frame = frame.shape[:2]

    pose_kpts = person_data.get('landmarks')  # [33, 4] normalized
    stick_endpoints = person_data.get('stick_endpoints')
    stick_detected = stick_endpoints is not None

    if pose_kpts is None or pose_kpts.shape[0] < 33:
        return None, None

    # Rebuild stick_kpts [2, 4] normalized
    if stick_detected:
        grip_pt, tip_pt = stick_endpoints
        stick_kpts = np.array([
            [grip_pt[0] / w_frame, grip_pt[1] / h_frame, 0.0, 1.0],
            [tip_pt[0] / w_frame, tip_pt[1] / h_frame, 0.0, 1.0]
        ], dtype=np.float32)
    else:
        stick_kpts = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    # Global features
    global_features = compute_global_features_from_kpts(
        pose_kpts, stick_kpts,
        has_stick_detected=stick_detected,
        version='v5'
    )

    # ── Similarity snapshot (target) — secondary "form quality" metric ──
    similarity_data = gcn_engine.capture_similarity_snapshot(
        pose_kpts, stick_kpts, global_features,
        target_class=target_pose,
        approach='simple'
    )
    form_quality = similarity_data['actual_score']      # 0-100
    display_quality = similarity_data['display_score']
    low_features = similarity_data['low_features']

    # ── Similarity snapshot (opposite) — symmetry diagnosis ──
    opposite_class = OPPOSITE_CLASS.get(target_pose)
    opposite_score = None
    if opposite_class:
        opp_data = gcn_engine.capture_similarity_snapshot(
            pose_kpts, stick_kpts, global_features,
            target_class=opposite_class,
            approach='simple'
        )
        opposite_score = opp_data['actual_score']

    # ── Tips from template low-features ──
    corrections = gcn_engine.get_feature_corrections(global_features, target_pose)
    if corrections:
        raw_values = corrections.get('raw_values', {})
        template_means = corrections.get('template_means', {})
        tips = generate_lesson_tips(low_features, raw_values, template_means, max_tips=3)
    else:
        tips = []

    # ── PRIMARY LESSON STATUS: classifier-driven ──
    predicted_class = person_data.get('predicted_class', 'N/A')
    classifier_conf = person_data.get('confidence', 0.0)
    status, status_color = get_lesson_status(predicted_class, target_pose, classifier_conf)

    annotated = draw_similarity_overlay(
        frame.copy(), target_pose, form_quality, display_quality,
        status, status_color, tips, stick_detected,
        predicted_class, classifier_conf,
        opposite_score, opposite_class,
    )

    result = {
        'form_quality': form_quality,
        'display_quality': display_quality,
        'status': status,
        'passed': similarity_data['passed'],
        'stick_detected': stick_detected,
        'predicted_class': predicted_class,
        'classifier_conf': classifier_conf,
        'tips': tips,
        'low_features': low_features,
        'opposite_class': opposite_class,
        'opposite_score': opposite_score,
    }
    return annotated, result


def _print_person_result(idx, result):
    print(f"\n  Person {idx}:")
    print(f"    Detected:      {result['predicted_class']} @ {result['classifier_conf']*100:.1f}%")
    print(f"    Status:        {result['status']}")
    print(f"    Form Quality:  {result['form_quality']:.1f}%")
    if result.get('opposite_class') and result.get('opposite_score') is not None:
        opp = result['opposite_class'].replace('_correct', '').replace('_', ' ').title()
        flag = " ⚠ OPPOSITE HIGHER" if result['opposite_score'] > result['form_quality'] else ""
        print(f"    Opposite:      {result['opposite_score']:.1f}%  ({opp}){flag}")
    print(f"    Stick:         {'Yes' if result['stick_detected'] else 'No'}")
    if result['tips']:
        for i, tip in enumerate(result['tips'], 1):
            print(f"    Tip {i}:        {tip}")


def process_single_image(image_path, target_pose, viewpoint, outdir=None,
                         interactive=True, pose_analyzer=None,
                         gcn_engine=None):
    """Process one image. Returns list of (annotated, result) tuples.
    If interactive, shows the first person in a window.
    """
    if not os.path.exists(image_path):
        print(f"[✗] Image not found: {image_path}")
        return []

    print(f"\n[LOAD] {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[✗] Failed to load image: {image_path}")
        return []

    h_frame, w_frame = frame.shape[:2]
    print(f"  Resolution: {w_frame}x{h_frame}")

    # ── Lazy init analyzers (shared across batch) ──
    close_on_exit = False
    if pose_analyzer is None:
        print("[INIT] Loading PoseAnalyzer...")
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=False
        )
        gcn_engine = pose_analyzer.gcn_engine
        if gcn_engine is None:
            print("[✗] GCN Engine not available")
            return []
        close_on_exit = True

    if gcn_engine is None:
        gcn_engine = pose_analyzer.gcn_engine
    gcn_engine.set_viewpoint(viewpoint)

    # ── Detect ──
    print("[DETECT] Running pose + stick detection...")
    results = pose_analyzer.process_frame(frame, skip_ml_inference=False, mode='snapshot')
    if not results or len(results) == 0:
        print("[✗] No person detected")
        return []

    print(f"[DETECT] {len(results)} person(s) found")

    all_outputs = []
    for pidx, person_data in enumerate(results):
        annotated, result = _compute_person_similarity(
            person_data, frame, target_pose, viewpoint, gcn_engine
        )
        if result is None:
            print(f"  [WARN] Person {pidx}: could not compute similarity")
            continue
        _print_person_result(pidx, result)
        all_outputs.append((annotated, result, pidx))

    # ── Save ──
    stem = Path(image_path).stem
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = "."

    for annotated, result, pidx in all_outputs:
        suffix = f"_p{pidx}" if len(all_outputs) > 1 else ""
        out_name = f"lesson_similarity_{stem}{suffix}.jpg"
        out_path = os.path.join(outdir, out_name)
        cv2.imwrite(out_path, annotated)
        print(f"  [SAVE] {out_path}")

    # ── Interactive window (first person only) ──
    if interactive and all_outputs:
        cv2.imshow("Lesson Similarity Result", all_outputs[0][0])
        print("\nPress any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if close_on_exit:
        pose_analyzer.close()

    return all_outputs


def batch_test(image_paths, target_pose, viewpoint, outdir):
    """Non-interactive batch processing."""
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*70}")
    print("BATCH LESSON SIMILARITY TEST")
    print(f"{'='*70}")
    print(f"Images:     {len(image_paths)}")
    print(f"Target:     {target_pose}")
    print(f"Viewpoint:  {viewpoint}")
    print(f"Output:     {outdir}")
    print(f"{'='*70}\n")

    # Shared analyzers
    print("[INIT] Loading models (shared across batch)...")
    stick_model_path = get_resource_path('deployment_package/weights/best.pt')
    pose_analyzer = PoseAnalyzer(
        detection_interval=3,
        stick_model_path=stick_model_path,
        debug_stick=False
    )
    gcn_engine = pose_analyzer.gcn_engine
    if gcn_engine is None:
        print("[✗] GCN Engine not available")
        return

    summary = []
    for idx, path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] {path}")
        outputs = process_single_image(
            path, target_pose, viewpoint,
            outdir=outdir, interactive=False,
            pose_analyzer=pose_analyzer, gcn_engine=gcn_engine
        )
        if outputs:
            for _, result, pidx in outputs:
                summary.append((
                    os.path.basename(path),
                    pidx,
                    result['predicted_class'],
                    result['classifier_conf'],
                    result['form_quality'],
                    result.get('opposite_score'),
                    result['status'],
                    result['stick_detected'],
                ))
        else:
            summary.append((os.path.basename(path), 0, "N/A", 0.0, 0.0, None, "NO DETECTION", False))

    # Summary table
    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print(f"{'='*70}")
    header = f"{'Image':<25} | {'P':>2} | {'Detected':<22} | {'D-Conf':>6} | {'Form%':>6} | {'Opp%':>6} | {'Status':<12} | {'Stick':>5}"
    print(header)
    print("-" * len(header))
    for row in summary:
        if len(row) == 8:
            name, pidx, pred, dconf, fq, opp, status, stick = row
            opp_str = f"{opp:>6.1f}" if opp is not None else "   N/A"
        else:
            name, pidx, pred, dconf, fq, status, stick = row
            opp_str = "   N/A"
        print(f"{name:<25} | {pidx:>2} | {pred:<22} | {dconf*100:>6.1f} | {fq:>6.1f} | {opp_str} | {status:<12} | {'Y' if stick else 'N':>5}")
    print(f"{'='*70}")
    print(f"Done. {len(image_paths)} image(s) processed. Results in: {outdir}\n")

    pose_analyzer.close()


def _collect_image_paths(inputs):
    """Expand folders and globs into a flat list of image paths."""
    collected = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                collected.extend(glob_module.glob(os.path.join(item, ext)))
        else:
            expanded = glob_module.glob(item)
            if expanded:
                collected.extend(expanded)
            elif os.path.exists(item):
                collected.append(item)
    # De-duplicate while preserving order
    seen = set()
    out = []
    for p in collected:
        p = os.path.abspath(p)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse_args(argv):
    """Manual parser supporting both single-image and --batch modes."""
    valid_targets = {
        'crown_thrust_correct', 'left_chest_thrust_correct',
        'left_elbow_block_correct', 'left_eye_thrust_correct',
        'left_knee_block_correct', 'left_temple_block_correct',
        'right_chest_thrust_correct', 'right_elbow_block_correct',
        'right_eye_thrust_correct', 'right_knee_block_correct',
        'right_temple_block_correct', 'solar_plexus_thrust_correct',
        'neutral'
    }

    if '--batch' in argv:
        batch_idx = argv.index('--batch')
        inputs = []
        i = batch_idx + 1
        while i < len(argv) and not argv[i].startswith('--'):
            inputs.append(argv[i])
            i += 1

        target_pose = None
        viewpoint = 'front'
        outdir = 'lesson_similarity_results'

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

        if target_pose is None:
            print("[✗] Batch mode requires --target <technique>")
            sys.exit(1)
        if target_pose not in valid_targets:
            print(f"[⚠] Unknown target '{target_pose}', proceeding anyway.")

        image_paths = _collect_image_paths(inputs)
        if not image_paths:
            print("[✗] No images found for batch processing.")
            sys.exit(1)

        return 'batch', image_paths, target_pose, viewpoint, outdir

    # ── Single-image backward-compatible mode ──
    if len(argv) < 3:
        return 'help', [], None, 'front', '.'

    image_path = argv[1]
    target_pose = argv[2]
    viewpoint = argv[3].lower() if len(argv) > 3 else 'front'

    if target_pose not in valid_targets:
        print(f"[⚠] Unknown target '{target_pose}', proceeding anyway.")
    if viewpoint not in ('front', 'left', 'right'):
        print(f"[⚠] Unknown viewpoint '{viewpoint}', using 'front'")
        viewpoint = 'front'

    return 'single', [image_path], target_pose, viewpoint, '.'


def print_help():
    print(__doc__)
    print("\nValid techniques for --target:")
    for t in sorted({
        'crown_thrust_correct', 'left_chest_thrust_correct',
        'left_elbow_block_correct', 'left_eye_thrust_correct',
        'left_knee_block_correct', 'left_temple_block_correct',
        'right_chest_thrust_correct', 'right_elbow_block_correct',
        'right_eye_thrust_correct', 'right_knee_block_correct',
        'right_temple_block_correct', 'solar_plexus_thrust_correct',
        'neutral'
    }):
        print(f"  - {t}")


def main():
    mode, image_paths, target_pose, viewpoint, outdir = parse_args(sys.argv)

    if mode == 'help':
        print_help()
        sys.exit(1)

    if mode == 'batch':
        batch_test(image_paths, target_pose, viewpoint, outdir)
    else:
        process_single_image(
            image_paths[0], target_pose, viewpoint,
            outdir=outdir, interactive=True
        )


if __name__ == '__main__':
    main()
