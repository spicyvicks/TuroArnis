"""
All-in-One Batch Test: All Viewpoints + All Classes in Lesson Mode

Walks test_images/ recursively, skipping 'neutral' folders.
For every image, runs lesson-mode classification (target = folder name,
viewpoint = parent folder) using the new V6 MultiViewpointEngine.

Usage:
    python scripts/batch_test_all_viewpoints_lesson.py

Output:
    - results/batch_test/annotated/  (overlaid images)
    - results/batch_test/result_table.json
    - Summary printed to stdout
"""

import cv2
import numpy as np
import sys
import os
import json
import time
from collections import defaultdict, Counter
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.utils.resource_path import get_resource_path

# Import test_classification's run_inference
sys.path.insert(0, os.path.join(project_root, 'app'))
from test_classification import run_inference


# ── CONFIG ───────────────────────────────────────────────────────────

INPUT_ROOT = os.path.join(project_root, 'test_images')
OUTPUT_DIR = os.path.join(project_root, 'results', 'batch_test_v5')
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, 'annotated')
JSON_PATH = os.path.join(OUTPUT_DIR, 'result_table.json')

# Extensions to process
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    # Initialize once, switch viewpoint per image
    print("=" * 70)
    print("V6 ALL-VIEWPOINT BATCH TEST — LESSON MODE")
    print("=" * 70)

    stick_model_path = get_resource_path('deployment_package/weights/best.pt')
    print(f"[1] Initializing PoseAnalyzer...")
    pose_analyzer = PoseAnalyzer(
        detection_interval=3,
        stick_model_path=stick_model_path,
        debug_stick=False
    )
    feedback_analyzer = FeedbackAnalyzer()

    if not pose_analyzer.gcn_engine:
        print("[✗] GCN Engine not available. Aborting.")
        return
    print("[✓] PoseAnalyzer + GCN ready.\n")

    # Collect all images
    tasks = []  # list of (image_path, viewpoint, target_class, rel_subdir)
    for viewpoint in ('front', 'left', 'right'):
        vp_dir = os.path.join(INPUT_ROOT, viewpoint)
        if not os.path.isdir(vp_dir):
            continue
        for class_name in sorted(os.listdir(vp_dir)):
            if class_name.lower() == 'neutral':
                continue
            class_dir = os.path.join(vp_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                img_path = os.path.join(class_dir, fname)
                rel = os.path.join(viewpoint, class_name, fname)
                tasks.append((img_path, viewpoint, class_name, rel))

    total = len(tasks)
    print(f"[2] Found {total} images to test (skipped 'neutral')\n")

    # ── Stats containers ──────────────────────────────────────────
    # Per viewpoint x class
    per_vc = defaultdict(lambda: {'count': 0, 'correct': 0, 'confidences': []})
    # Per viewpoint overall
    per_vp = defaultdict(lambda: {'count': 0, 'correct': 0, 'confidences': []})
    # Confusion matrix: (viewpoint, target, predicted) -> count
    confusion = Counter()
    # Missed detections
    missed = Counter()

    start_all = time.time()

    for idx, (img_path, viewpoint, target_class, rel) in enumerate(tasks, 1):
        pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
        # Reset session cache for clean inference each image
        pose_analyzer.clear_session_cache()

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{idx}/{total}] SKIP (load fail): {rel}")
            continue

        try:
            annotated, predicted, confidence, status, fb, stick = run_inference(
                frame, target_pose=target_class, viewpoint=viewpoint,
                pose_analyzer=pose_analyzer, feedback_analyzer=feedback_analyzer
            )
        except Exception as e:
            print(f"[{idx}/{total}] SKIP (inference error): {rel} — {e}")
            continue

        # Save annotated overlay
        out_path = os.path.join(ANNOTATED_DIR, rel.replace('/', os.sep).replace('\\', os.sep))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, annotated)

        # Determine correctness
        is_correct = (predicted == target_class)
        if predicted == 'No Technique Detected' or predicted == 'neutral':
            is_correct = False
            missed[(viewpoint, target_class)] += 1

        # Record stats
        vc_key = (viewpoint, target_class)
        per_vc[vc_key]['count'] += 1
        per_vc[vc_key]['correct'] += int(is_correct)
        per_vc[vc_key]['confidences'].append(confidence)

        per_vp[viewpoint]['count'] += 1
        per_vp[viewpoint]['correct'] += int(is_correct)
        per_vp[viewpoint]['confidences'].append(confidence)

        confusion[(viewpoint, target_class, predicted)] += 1

        marker = "✓" if is_correct else "✗"
        if idx % 20 == 0 or idx == total:
            print(f"[{idx}/{total}] {marker} {rel} | pred={predicted} | conf={confidence:.3f} | status={status}")

    elapsed = time.time() - start_all

    # ── BUILD RESULT TABLE ───────────────────────────────────────
    results = {
        'total_images': total,
        'elapsed_seconds': round(elapsed, 2),
        'per_viewpoint_class': [],
        'per_viewpoint': [],
        'overall': {},
        'confusion_matrix': [],
    }

    total_correct = 0
    total_count = 0
    all_confidences = []

    # Per viewpoint x class
    for (vp, cls) in sorted(per_vc.keys()):
        stats = per_vc[(vp, cls)]
        c = stats['count']
        corr = stats['correct']
        acc = corr / c * 100 if c else 0
        avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
        results['per_viewpoint_class'].append({
            'viewpoint': vp,
            'class': cls,
            'images': c,
            'correct': corr,
            'accuracy': round(acc, 2),
            'avg_confidence': round(float(avg_conf), 4),
        })
        total_correct += corr
        total_count += c
        all_confidences.extend(stats['confidences'])

    # Per viewpoint
    for vp in sorted(per_vp.keys()):
        stats = per_vp[vp]
        c = stats['count']
        corr = stats['correct']
        acc = corr / c * 100 if c else 0
        avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
        results['per_viewpoint'].append({
            'viewpoint': vp,
            'images': c,
            'correct': corr,
            'accuracy': round(acc, 2),
            'avg_confidence': round(float(avg_conf), 4),
        })

    # Overall
    results['overall'] = {
        'images': total_count,
        'correct': total_correct,
        'accuracy': round(total_correct / total_count * 100, 2) if total_count else 0,
        'avg_confidence': round(float(np.mean(all_confidences)), 4) if all_confidences else 0,
        'elapsed_seconds': round(elapsed, 2),
    }

    # Confusion matrix (top misclassifications only)
    for (vp, tgt, pred), cnt in confusion.most_common():
        if tgt != pred:
            results['confusion_matrix'].append({
                'viewpoint': vp,
                'target': tgt,
                'predicted': pred,
                'count': cnt,
            })

    with open(JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    # ── PRINT SUMMARY ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nOverall: {total_correct}/{total_count} correct = {results['overall']['accuracy']}%")
    print(f"Avg confidence: {results['overall']['avg_confidence']:.4f}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/total_count:.2f}s per image)\n")

    # Per-viewpoint
    print("-" * 70)
    print(f"{'Viewpoint':<12} {'Images':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Conf':>10}")
    print("-" * 70)
    for row in results['per_viewpoint']:
        print(f"{row['viewpoint']:<12} {row['images']:>8} {row['correct']:>8} {row['accuracy']:>9.2f}% {row['avg_confidence']:>10.4f}")
    print("-" * 70)

    # Per-viewpoint x class
    print("\nPer-Class Results:")
    print("-" * 80)
    print(f"{'Viewpoint':<10} {'Class':<35} {'Images':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Conf':>10}")
    print("-" * 80)
    for row in results['per_viewpoint_class']:
        cls_short = row['class'].replace('_correct', '')[:34]
        print(f"{row['viewpoint']:<10} {cls_short:<35} {row['images']:>8} {row['correct']:>8} {row['accuracy']:>9.2f}% {row['avg_confidence']:>10.4f}")
    print("-" * 80)

    # Top misclassifications
    print("\nTop Misclassifications (target → predicted):")
    print("-" * 70)
    miscount = 0
    for row in results['confusion_matrix'][:20]:
        if miscount >= 15:
            break
        print(f"  {row['viewpoint']:<8} {row['target']:<30} → {row['predicted']:<30} ({row['count']}x)")
        miscount += 1
    if not results['confusion_matrix']:
        print("  (none — perfect classification!)")
    print("-" * 70)

    # Missed detections summary
    if missed:
        print("\nMissed Detections (No Technique Detected / neutral):")
        for (vp, cls), cnt in missed.most_common(10):
            print(f"  {vp:<8} {cls:<35} — {cnt} images")

    print(f"\n[✓] Results saved to: {JSON_PATH}")
    print(f"[✓] Annotated images saved to: {ANNOTATED_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
