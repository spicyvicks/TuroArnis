"""
test_images_batch.py
====================
CLI batch tester — runs one or more images through the full TuroArnis
analysis pipeline (GCN + FeedbackAnalyzer) without any GUI/splash screens.

Usage examples
--------------
# Single image, auto-detect viewpoint (front)
  python test_images_batch.py path/to/image.jpg

# Multiple images
  python test_images_batch.py img1.jpg img2.jpg img3.jpg

# Glob / directory
  python test_images_batch.py "data/test_images/*.jpg"

# Override viewpoint (front / left / right) and target pose
  python test_images_batch.py img.jpg --viewpoint right --target crown_thrust_correct

# Save results to a JSON file
  python test_images_batch.py *.jpg --json results.json

# Suppress verbose debug output from YOLO / MediaPipe / GCN internals
  python test_images_batch.py img.jpg --quiet
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# ── ensure project root is on the path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

# Suppress noisy TF / YOLO startup logs when --quiet is active
# (we patch this after arg-parse below)

# ── imports from the project ──────────────────────────────────────────────────
from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.models.gcn.model_architecture import CLASS_NAMES
from app.utils.resource_path import get_resource_path


# ── colour helpers for terminal output ────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def _confidence_color(confidence: float, threshold: float) -> str:
    if confidence >= threshold + 0.15:
        return GREEN
    if confidence >= threshold:
        return YELLOW
    return RED


def _band_label(confidence: float, threshold: float) -> str:
    if confidence >= threshold + 0.15:
        return "EXCELLENT"
    if confidence >= threshold:
        return "GOOD"
    return "BELOW THRESHOLD"


def _divider(char="─", width=70):
    print(char * width)


# ── core analysis for one image ───────────────────────────────────────────────
def analyze_image(
    image_path: str,
    pose_analyzer: PoseAnalyzer,
    feedback_analyzer: FeedbackAnalyzer,
    viewpoint: str = "front",
    target_pose: str = None,
    quiet: bool = False,
) -> dict:
    """
    Run the full pipeline on a single image and return a result dict.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"image": image_path, "error": "Could not read image"}

    h, w = img.shape[:2]

    # Set GCN viewpoint
    pose_analyzer.set_viewpoint(viewpoint)
    pose_analyzer.clear_session_cache()

    # Run snapshot analysis
    t0 = time.perf_counter()
    raw_results = pose_analyzer.process_frame(
        img,
        skip_ml_inference=False,
        skip_stick_detection=False,
        mode="snapshot",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Grab the first person result (zone 0)
    zone_result = raw_results[0] if raw_results else {}

    predicted_class = zone_result.get("predicted_class", "N/A")
    confidence      = zone_result.get("confidence", 0.0)
    stick_detected  = zone_result.get("stick_endpoints") is not None
    global_features = zone_result.get("global_features")
    live_angles     = zone_result.get("live_angles") or {}

    # Per-viewpoint threshold
    gcn_config  = pose_analyzer.gcn_engine.config if pose_analyzer.gcn_engine else {}
    threshold   = gcn_config.get("models", {}).get(viewpoint, {}).get("confidence_threshold", 0.55)

    # ── per-class softmax probabilities ──────────────────────────────────────
    # Re-run inference to collect all_probabilities (the main predict() call
    # above already does the multi-hypothesis loop; here we expose its result).
    all_probs = {}
    if pose_analyzer.gcn_engine and global_features:
        try:
            from app.models.gcn.feature_extraction import (
                extract_node_features,
                compute_hybrid_features,
            )
            import torch
            engine = pose_analyzer.gcn_engine
            model  = engine.models.get(engine.current_viewpoint)
            if model:
                # We need pose_keypoints and stick_keypoints; reconstruct from zone_result
                lm2d = zone_result.get("landmarks")
                stick_ep = zone_result.get("stick_endpoints")
                if lm2d is not None:
                    lm_list = lm2d.landmark
                    pose_kpts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lm_list])
                    if stick_ep:
                        grip_pt, tip_pt = stick_ep
                        stick_kpts = np.array([
                            [grip_pt[0]/w, grip_pt[1]/h, 0.0, 1.0],
                            [tip_pt[0]/w,  tip_pt[1]/h,  0.0, 1.0],
                        ], dtype=np.float32)
                    else:
                        stick_kpts = np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]], dtype=np.float32)

                    node_features = extract_node_features(pose_kpts, stick_kpts)
                    x       = torch.tensor(node_features, dtype=torch.float32).to(engine.device)
                    batch   = torch.zeros(35, dtype=torch.long).to(engine.device)

                    best_probs   = None
                    best_class_c = None
                    best_conf_c  = 0.0
                    import torch.nn.functional as F
                    with torch.no_grad():
                        candidates = [c for c in CLASS_NAMES if c != "neutral"]
                        for candidate in candidates:
                            hf = compute_hybrid_features(
                                global_features, engine.templates,
                                viewpoint=viewpoint, class_name=candidate,
                            )
                            h_t   = torch.tensor(hf, dtype=torch.float32).unsqueeze(0).to(engine.device)
                            logits = model(x, engine.edge_index, batch, h_t)
                            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
                            c_idx  = CLASS_NAMES.index(candidate)
                            c_conf = float(probs[c_idx])
                            if c_conf > best_conf_c:
                                best_conf_c  = c_conf
                                best_class_c = candidate
                                best_probs   = probs

                    if best_probs is not None:
                        all_probs = {CLASS_NAMES[i]: float(best_probs[i]) for i in range(len(CLASS_NAMES))}
        except Exception as e:
            if not quiet:
                print(f"  [WARN] Could not extract per-class probabilities: {e}")

    # ── feedback analysis ─────────────────────────────────────────────────────
    effective_target = target_pose or predicted_class
    analysis = {}
    feedback_messages = []
    if feedback_analyzer and effective_target not in ("N/A", "No Technique Detected", ""):
        zone_result["frame_w"] = w
        zone_result["frame_h"] = h
        analysis = feedback_analyzer.analyze(
            result=zone_result,
            target_form=effective_target,
            confidence_threshold=threshold,
            viewpoint=viewpoint,
            gcn_engine=pose_analyzer.gcn_engine,
        )
        prioritized = feedback_analyzer.get_prioritized_messages(analysis, max_messages=10)
        feedback_messages = [{"message": msg, "type": t} for msg, t in prioritized]

    return {
        "image":            image_path,
        "viewpoint":        viewpoint,
        "elapsed_ms":       round(elapsed_ms, 1),
        "predicted_class":  predicted_class,
        "confidence":       round(confidence, 4),
        "threshold":        round(threshold, 4),
        "band":             _band_label(confidence, threshold),
        "target_pose":      effective_target,
        "is_correct":       analysis.get("is_correct", False),
        "stick_detected":   stick_detected,
        "severity":         analysis.get("severity", "n/a"),
        "errors":           analysis.get("errors", []),
        "warnings":         analysis.get("warnings", []),
        "suggestions":      analysis.get("suggestions", []),
        "feedback_messages": feedback_messages,
        "live_angles":      live_angles,
        "all_probs":        all_probs,
    }


# ── pretty-print one result ───────────────────────────────────────────────────
def print_result(r: dict, idx: int, total: int, show_probs: bool = True):
    _divider("═")
    img_name  = Path(r["image"]).name
    print(f"{BOLD}[{idx}/{total}] {img_name}{RESET}  {DIM}({r['elapsed_ms']} ms){RESET}")
    _divider()

    if "error" in r:
        print(f"  {RED}ERROR: {r['error']}{RESET}")
        return

    vp    = r["viewpoint"]
    pred  = r["predicted_class"]
    conf  = r["confidence"]
    thr   = r["threshold"]
    band  = r["band"]
    cc    = _confidence_color(conf, thr)

    print(f"  Viewpoint     : {CYAN}{vp}{RESET}")
    print(f"  Detected      : {BOLD}{pred}{RESET}")
    print(f"  Confidence    : {cc}{conf:.1%}{RESET}  [{cc}{band}{RESET}]  (threshold={thr:.0%})")
    print(f"  Stick         : {'✓ detected' if r['stick_detected'] else '✗ not detected'}")
    print(f"  Target pose   : {r['target_pose']}")
    print(f"  Is correct    : {'✓ YES' if r['is_correct'] else '✗ NO'}  (severity: {r['severity']})")

    # ── per-class softmax table ───────────────────────────────────────────────
    if show_probs and r["all_probs"]:
        print()
        print(f"  {BOLD}Softmax Probabilities (all classes):{RESET}")
        sorted_probs = sorted(r["all_probs"].items(), key=lambda x: -x[1])
        for cls, p in sorted_probs:
            bar    = "█" * int(p * 30)
            marker = " ← predicted" if cls == pred else ""
            color  = GREEN if cls == pred else RESET
            print(f"    {color}{cls:<35}{RESET} {p:6.1%}  {DIM}{bar}{RESET}{marker}")

    # ── live joint angles ─────────────────────────────────────────────────────
    if r.get("live_angles"):
        print()
        print(f"  {BOLD}Live Joint Angles:{RESET}")
        for joint, angle in r["live_angles"].items():
            print(f"    {joint:<25} {angle:6.1f}°")

    # ── feedback messages ─────────────────────────────────────────────────────
    print()
    if r["feedback_messages"]:
        print(f"  {BOLD}Feedback / Corrections:{RESET}")
        TYPE_COLORS = {"error": RED, "warning": YELLOW, "suggestion": CYAN}
        for item in r["feedback_messages"]:
            tc = TYPE_COLORS.get(item["type"], RESET)
            print(f"    {tc}[{item['type'].upper()}]{RESET}  {item['message']}")
    else:
        print(f"  {GREEN}No corrections — form looks good!{RESET}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch-test TuroArnis pose analysis on images (no GUI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images", nargs="+",
        help="Image file(s) or glob patterns (e.g. data/*.jpg)"
    )
    parser.add_argument(
        "--viewpoint", "-v", default="front",
        choices=["front", "left", "right"],
        help="Camera viewpoint for the GCN model (default: front)"
    )
    parser.add_argument(
        "--target", "-t", default=None,
        metavar="POSE_KEY",
        help="Target pose key for feedback (e.g. crown_thrust_correct). "
             "Defaults to whichever pose the model predicts."
    )
    parser.add_argument(
        "--json", "-j", default=None,
        metavar="FILE",
        help="Save all results to a JSON file."
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress internal debug logs (YOLO / MediaPipe / GCN)."
    )
    parser.add_argument(
        "--no-probs", action="store_true",
        help="Skip printing per-class softmax probability table."
    )
    parser.add_argument(
        "--stick-model", default=None,
        metavar="PATH",
        help="Path to YOLO stick-detection model (optional, improves stick feedback)."
    )
    args = parser.parse_args()

    # ── resolve image list (supports globs) ───────────────────────────────────
    image_paths = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        elif os.path.isfile(pattern):
            image_paths.append(pattern)
        else:
            print(f"[WARN] No files matched: {pattern}")
    image_paths = [str(Path(p).resolve()) for p in image_paths]

    if not image_paths:
        print("[ERROR] No valid image files found. Exiting.")
        sys.exit(1)

    # ── suppress noisy internal logs if --quiet ───────────────────────────────
    if args.quiet:
        import logging
        logging.disable(logging.WARNING)
        os.environ["YOLO_VERBOSE"] = "False"
        # Redirect stdout during init if needed; we leave that for a future improvement.

    # ── initialise pipeline ───────────────────────────────────────────────────
    print(f"\n{BOLD}TuroArnis — Image Batch Tester{RESET}")
    _divider()

    stick_model_path = args.stick_model
    if stick_model_path is None:
        # Try to find the default stick model relative to the project
        candidate = get_resource_path("app/models/stick_model/best.pt")
        if os.path.exists(candidate):
            stick_model_path = candidate
            print(f"[INFO] Auto-detected stick model: {candidate}")

    print("[INFO] Initialising PoseAnalyzer...")
    pose_analyzer = PoseAnalyzer(
        stick_model_path=stick_model_path,
        debug_stick=not args.quiet,
    )

    print("[INFO] Initialising FeedbackAnalyzer...")
    feedback_analyzer = FeedbackAnalyzer()

    _divider()
    print(f"[INFO] {len(image_paths)} image(s) queued | viewpoint={args.viewpoint}")
    if args.target:
        print(f"[INFO] Fixed target pose: {args.target}")
    _divider()

    # ── run batch ─────────────────────────────────────────────────────────────
    all_results = []
    for idx, img_path in enumerate(image_paths, start=1):
        result = analyze_image(
            image_path=img_path,
            pose_analyzer=pose_analyzer,
            feedback_analyzer=feedback_analyzer,
            viewpoint=args.viewpoint,
            target_pose=args.target,
            quiet=args.quiet,
        )
        all_results.append(result)
        print_result(result, idx=idx, total=len(image_paths), show_probs=not args.no_probs)

    # ── summary ───────────────────────────────────────────────────────────────
    _divider("═")
    print(f"{BOLD}SUMMARY — {len(all_results)} image(s){RESET}")
    _divider()
    detected  = [r for r in all_results if r.get("predicted_class") not in ("N/A", "No Technique Detected")]
    correct   = [r for r in all_results if r.get("is_correct")]
    excellent = [r for r in all_results if r.get("band") == "EXCELLENT"]
    avg_conf  = (
        sum(r["confidence"] for r in detected) / len(detected)
        if detected else 0.0
    )

    print(f"  Detected   : {len(detected)}/{len(all_results)}")
    print(f"  Correct    : {len(correct)}/{len(all_results)}")
    print(f"  Excellent  : {len(excellent)}/{len(all_results)}")
    print(f"  Avg Conf   : {avg_conf:.1%}")

    # per-class breakdown
    from collections import Counter
    class_counts = Counter(r["predicted_class"] for r in all_results)
    print()
    print(f"  {BOLD}Detected class breakdown:{RESET}")
    for cls, cnt in class_counts.most_common():
        print(f"    {cls:<40} × {cnt}")
    _divider()

    # ── optional JSON export ───────────────────────────────────────────────────
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Results saved to: {args.json}")

    pose_analyzer.close()


if __name__ == "__main__":
    main()
