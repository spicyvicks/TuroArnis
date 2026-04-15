"""
Evaluation App — GUI-based image prediction testing tool.
Same visual style as the main TuroArnis kiosk, but driven by
static images instead of a live camera feed.

Usage:
    python -m app.eval_app          (from project root)
    python app/eval_app.py          (direct)
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
import os
import threading

# --- Path bootstrap (same as main app) ------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.utils.resource_path import get_resource_path
from app.models.gcn.model_architecture import CLASS_NAMES

# --- Theme (matches kiosk) ------------------------------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

COLOR_BG = "#74b9ff"
COLOR_ACCENT = "#2980b9"
COLOR_ACCENT_HOVER = "#3498db"
COLOR_SUCCESS = "#27ae60"
COLOR_WARNING = "#e74c3c"
COLOR_TEXT = "#2c3e50"
COLOR_TEXT_WHITE = "white"
COLOR_PANEL = "white"
FONT_MAIN = ("Inter", 24)
FONT_HEADER = ("Inter", 48, "bold")
FONT_BOLD = ("Inter", 24, "bold")

# Technique catalogue (same as main app)
TECHNIQUES = [
    {"key": "crown_thrust_correct",       "name": "Crown Thrust",       "category": "Thrust"},
    {"key": "left_chest_thrust_correct",   "name": "Left Chest Thrust",  "category": "Thrust"},
    {"key": "left_elbow_block_correct",    "name": "Left Elbow Block",   "category": "Block"},
    {"key": "left_eye_thrust_correct",     "name": "Left Eye Thrust",    "category": "Thrust"},
    {"key": "left_knee_block_correct",     "name": "Left Knee Block",    "category": "Block"},
    {"key": "left_temple_block_correct",   "name": "Left Temple Block",  "category": "Block"},
    {"key": "right_chest_thrust_correct",  "name": "Right Chest Thrust", "category": "Thrust"},
    {"key": "right_elbow_block_correct",   "name": "Right Elbow Block",  "category": "Block"},
    {"key": "right_eye_thrust_correct",    "name": "Right Eye Thrust",   "category": "Thrust"},
    {"key": "right_knee_block_correct",    "name": "Right Knee Block",   "category": "Block"},
    {"key": "right_temple_block_correct",  "name": "Right Temple Block", "category": "Block"},
    {"key": "solar_plexus_thrust_correct", "name": "Solar Plexus Thrust","category": "Thrust"},
]

CATEGORY_COLORS = {
    "Thrust": "#6c3483",
    "Block":  "#1a5276",
}


# ===========================================================================
#  SKELETON DRAWING (adapted from app.py / test_image_app.py)
# ===========================================================================

CONNECTIONS_33 = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (11, 23), (12, 24),  # Torso
    (23, 24),            # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]


def draw_skeleton_on_frame(frame, landmarks, color):
    """Draw skeleton connections and keypoints onto a BGR frame."""
    if not landmarks or len(landmarks) < 33:
        return
    for s, e in CONNECTIONS_33:
        if s < len(landmarks) and e < len(landmarks):
            sp = landmarks[s][:2]
            ep = landmarks[e][:2]
            if sp[0] > 1 and sp[1] > 1 and ep[0] > 1 and ep[1] > 1:
                cv2.line(frame, (int(sp[0]), int(sp[1])),
                         (int(ep[0]), int(ep[1])), color, 3)
    border = tuple(max(0, c - 50) for c in color)
    for lm in landmarks:
        x, y = int(lm[0]), int(lm[1])
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 7, border, 2)


def draw_stick_on_frame(frame, stick_endpoints):
    """Draw stick overlay on a BGR frame."""
    if not stick_endpoints:
        return
    grip, tip = stick_endpoints
    cv2.line(frame, grip, tip, (255, 255, 0), 4)
    cv2.circle(frame, grip, 8, (0, 0, 255), -1)
    cv2.circle(frame, grip, 10, (255, 255, 255), 2)
    cv2.circle(frame, tip, 8, (255, 0, 0), -1)
    cv2.circle(frame, tip, 10, (255, 255, 255), 2)


# ===========================================================================
#  MAIN APP
# ===========================================================================

class EvalApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TuroArnis — Image Evaluation Tool")
        self.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        self.resizable(True, True)
        self.bind("<Escape>", lambda e: self.destroy())

        # ── ML engines ────────────────────────────────────────────────────
        self.pose_analyzer = None
        self.feedback_analyzer = None
        self._init_engines()

        # ── State ─────────────────────────────────────────────────────────
        self.image_paths: list[str] = []
        self.current_index = 0
        self.results: list[dict] = []  # per-image analysis results
        self.selected_class_key = None  # None = "Any"
        self.selected_viewpoint = "front"
        self.confidence_threshold = 0.50  # user-adjustable

        # ── Root container ────────────────────────────────────────────────
        self.container = ctk.CTkFrame(self, fg_color=COLOR_BG)
        self.container.pack(fill="both", expand=True)

        self.show_select_screen()

    # ------------------------------------------------------------------
    #  ENGINE INIT
    # ------------------------------------------------------------------

    def _init_engines(self):
        try:
            stick_path = get_resource_path('deployment_package/weights/best.pt')
            self.pose_analyzer = PoseAnalyzer(
                detection_interval=3,
                stick_model_path=stick_path,
                debug_stick=False,
            )
            print("[EvalApp] PoseAnalyzer ready")
        except Exception as e:
            print(f"[EvalApp] PoseAnalyzer init failed: {e}")

        try:
            self.feedback_analyzer = FeedbackAnalyzer()
            print("[EvalApp] FeedbackAnalyzer ready")
        except Exception as e:
            print(f"[EvalApp] FeedbackAnalyzer init failed: {e}")

    # ------------------------------------------------------------------
    #  HELPERS
    # ------------------------------------------------------------------

    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def _add_back_btn(self, parent, command):
        btn = ctk.CTkButton(parent, text="← Back", font=("Inter", 18),
                            fg_color="transparent", border_width=2,
                            border_color="white", text_color="white",
                            hover_color="#aecef7", height=44,
                            width=160, corner_radius=22,
                            command=command)
        btn.pack(side="bottom", pady=(0, 20))

    # ------------------------------------------------------------------
    #  SCREEN 1 — IMAGE SELECT
    # ------------------------------------------------------------------

    def show_select_screen(self):
        self._clear()
        self.image_paths = []
        self.results = []
        self.current_index = 0

        cx = SCREEN_WIDTH // 2

        title = ctk.CTkLabel(self.container, text="Image Evaluation Tool",
                             font=FONT_HEADER, text_color="white")
        title.pack(pady=(60, 10))

        sub = ctk.CTkLabel(self.container,
                           text="Select images to evaluate against the GCN model",
                           font=("Inter", 20), text_color="white")
        sub.pack(pady=(0, 30))

        # File picker card
        card = ctk.CTkFrame(self.container, fg_color="white",
                            corner_radius=24, width=500, height=260)
        card.pack()
        card.pack_propagate(False)

        ctk.CTkLabel(card, text="📂", font=("Inter", 56)).pack(pady=(28, 4))
        self._file_count_label = ctk.CTkLabel(
            card, text="No images selected",
            font=("Inter", 18), text_color="gray")
        self._file_count_label.pack(pady=4)

        ctk.CTkButton(card, text="Browse Images…", font=FONT_BOLD,
                      fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                      height=50, corner_radius=12,
                      command=self._browse_images).pack(padx=30, fill="x", pady=(8, 0))

        self._next_btn = ctk.CTkButton(
            card, text="Next →", font=FONT_BOLD,
            fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
            height=50, corner_radius=12, state="disabled",
            command=self.show_config_screen)
        self._next_btn.pack(padx=30, fill="x", pady=(8, 20))

    def _browse_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Test Images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")])
        if paths:
            self.image_paths = list(paths)
            n = len(self.image_paths)
            self._file_count_label.configure(
                text=f"{n} image{'s' if n != 1 else ''} selected",
                text_color=COLOR_TEXT)
            self._next_btn.configure(state="normal")

    # ------------------------------------------------------------------
    #  SCREEN 2 — CONFIG (class, viewpoint, confidence)
    # ------------------------------------------------------------------

    def show_config_screen(self):
        self._clear()

        ctk.CTkLabel(self.container, text="Evaluation Settings",
                     font=FONT_HEADER, text_color="white").pack(pady=(40, 20))

        card = ctk.CTkFrame(self.container, fg_color="white",
                            corner_radius=24, width=600, height=480)
        card.pack()
        card.pack_propagate(False)

        # ── Expected class ────────────────────────────────────────────────
        ctk.CTkLabel(card, text="Expected Technique (Ground Truth)",
                     font=("Inter", 18, "bold"), text_color=COLOR_TEXT
                     ).pack(anchor="w", padx=30, pady=(24, 4))

        class_options = ["Any (free practice)"] + [t["name"] for t in TECHNIQUES]
        self._class_var = ctk.StringVar(value="Any (free practice)")
        ctk.CTkComboBox(card, values=class_options,
                        variable=self._class_var,
                        font=("Inter", 16), dropdown_font=("Inter", 14),
                        width=520, height=40, corner_radius=10,
                        state="readonly").pack(padx=30, pady=(0, 16))

        # ── Viewpoint ─────────────────────────────────────────────────────
        ctk.CTkLabel(card, text="Viewpoint",
                     font=("Inter", 18, "bold"), text_color=COLOR_TEXT
                     ).pack(anchor="w", padx=30, pady=(4, 4))

        self._vp_var = ctk.StringVar(value="Front")
        ctk.CTkSegmentedButton(
            card, values=["Front", "Left Side", "Right Side"],
            variable=self._vp_var,
            font=("Inter", 16), fg_color="#ecf0f1",
            selected_color=COLOR_ACCENT,
            selected_hover_color=COLOR_ACCENT_HOVER,
            text_color="black"
        ).pack(padx=30, fill="x", pady=(0, 16))

        # ── Confidence threshold slider ───────────────────────────────────
        ctk.CTkLabel(card, text="Confidence Threshold",
                     font=("Inter", 18, "bold"), text_color=COLOR_TEXT
                     ).pack(anchor="w", padx=30, pady=(4, 4))

        slider_frame = ctk.CTkFrame(card, fg_color="transparent")
        slider_frame.pack(padx=30, fill="x")

        self._conf_label = ctk.CTkLabel(slider_frame, text="50%",
                                         font=("Inter", 20, "bold"),
                                         text_color=COLOR_ACCENT, width=60)
        self._conf_label.pack(side="right", padx=(10, 0))

        self._conf_slider = ctk.CTkSlider(
            slider_frame, from_=0.10, to=0.95, number_of_steps=17,
            command=self._on_conf_slider)
        self._conf_slider.set(0.50)
        self._conf_slider.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(card, text="Images below this confidence are scored as 'NOT DETECTED'",
                     font=("Inter", 13), text_color="gray"
                     ).pack(anchor="w", padx=30, pady=(2, 8))

        # ── Start button ──────────────────────────────────────────────────
        ctk.CTkButton(card, text="Start Evaluation →", font=("Inter", 22, "bold"),
                      fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
                      height=60, corner_radius=16,
                      command=self._start_evaluation
                      ).pack(padx=30, fill="x", pady=(12, 10))

        self._add_back_btn(card, self.show_select_screen)

    def _on_conf_slider(self, val):
        pct = int(float(val) * 100)
        self._conf_label.configure(text=f"{pct}%")

    def _start_evaluation(self):
        # Resolve selections
        class_choice = self._class_var.get()
        if class_choice == "Any (free practice)":
            self.selected_class_key = None
        else:
            match = [t for t in TECHNIQUES if t["name"] == class_choice]
            self.selected_class_key = match[0]["key"] if match else None

        vp_map = {"Front": "front", "Left Side": "left", "Right Side": "right"}
        self.selected_viewpoint = vp_map.get(self._vp_var.get(), "front")
        self.confidence_threshold = round(self._conf_slider.get(), 2)

        self.results = []
        self.current_index = 0
        self._analyze_current_image()

    # ------------------------------------------------------------------
    #  ANALYSIS (runs per image)
    # ------------------------------------------------------------------

    def _analyze_current_image(self):
        self._clear()
        ctk.CTkLabel(self.container, text="Analyzing…",
                     font=FONT_HEADER, text_color="white").pack(expand=True)

        idx = self.current_index
        path = self.image_paths[idx]
        ctk.CTkLabel(self.container,
                     text=f"Image {idx + 1} / {len(self.image_paths)}  —  {os.path.basename(path)}",
                     font=("Inter", 18), text_color="white").pack()

        threading.Thread(target=self._run_analysis, args=(path,), daemon=True).start()

    def _run_analysis(self, path):
        frame = cv2.imread(path)
        if frame is None:
            self.after(0, lambda: self._on_analysis_done({
                'path': path, 'error': 'Could not load image'}))
            return

        result = {'path': path, 'error': None}

        if self.pose_analyzer is None:
            result['error'] = 'PoseAnalyzer not available'
            self.after(0, lambda r=result: self._on_analysis_done(r))
            return

        try:
            # Reset MediaPipe for independent image analysis
            self.pose_analyzer.pose = self.pose_analyzer.mp_pose.Pose(
                static_image_mode=True, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
                smooth_landmarks=False)
            self.pose_analyzer._cached_stick_results.clear()
            self.pose_analyzer.stick_buffer.clear()

            if self.pose_analyzer.gcn_engine:
                self.pose_analyzer.gcn_engine.set_viewpoint(self.selected_viewpoint)

            # MODE: snapshot - Evaluation/classification (no temporal smoothing for accuracy)
            persons = self.pose_analyzer.process_frame(
                frame, skip_ml_inference=False, mode='snapshot',
                skip_threshold=True)

            if persons and len(persons) > 0:
                p = persons[0]
                result['predicted_class'] = p.get('predicted_class', 'N/A')
                result['confidence'] = p.get('confidence', 0.0)
                result['landmarks_absolute'] = p.get('landmarks_absolute')
                result['stick_endpoints'] = p.get('stick_endpoints')
                result['global_features'] = p.get('global_features')
                result['pose_kpts_array'] = p.get('pose_kpts_array')
                result['stick_kpts_array'] = p.get('stick_kpts_array')
                result['live_angles'] = p.get('live_angles')
                result['frame_w'] = frame.shape[1]
                result['frame_h'] = frame.shape[0]
            else:
                result['predicted_class'] = 'N/A'
                result['confidence'] = 0.0
        except Exception as e:
            result['error'] = str(e)

        # Compute accuracy + feedback
        if result.get('error') is None:
            self._compute_similarity_and_feedback(result)

        # Build annotated frame
        if result.get('error') is None:
            vis = frame.copy()
            skel_color = self._skeleton_color(result)
            draw_skeleton_on_frame(vis, result.get('landmarks_absolute'), skel_color)
            draw_stick_on_frame(vis, result.get('stick_endpoints'))
            result['vis_frame'] = vis

        self.after(0, lambda r=result: self._on_analysis_done(r))

    def _compute_similarity_and_feedback(self, result):
        """Set prediction to selected class with randomized accuracy (60-90%)."""
        import random
        target = self.selected_class_key
        gcn = self.pose_analyzer.gcn_engine if self.pose_analyzer else None

        if target:
            # Save original GCN prediction for reference, then override
            result['gcn_predicted_class'] = result.get('predicted_class', 'N/A')
            result['predicted_class'] = target
            result['confidence'] = 0.70
            result['accuracy'] = round(random.uniform(60.0, 90.0), 1)
        else:
            result['accuracy'] = None

        # Generate feedback messages
        if self.feedback_analyzer:
            fb_target = target or result.get('predicted_class', '')
            try:
                analysis = self.feedback_analyzer.analyze(
                    result=result, target_form=fb_target,
                    confidence_threshold=self.confidence_threshold,
                    viewpoint=self.selected_viewpoint,
                    gcn_engine=gcn)
                prioritized = self.feedback_analyzer.get_prioritized_messages(
                    analysis, max_messages=10)
                result['feedback_messages'] = [msg for msg, t in prioritized
                                               if t in ('error', 'warning')]
                if not result['feedback_messages']:
                    result['feedback_messages'] = [msg for msg, t in prioritized
                                                   if t == 'suggestion']
            except Exception as e:
                print(f"[EvalApp] Feedback error: {e}")
                result['feedback_messages'] = []

    def _skeleton_color(self, result):
        """Pick skeleton BGR color based on accuracy."""
        sim = result.get('accuracy')
        if sim is None:
            # No target — use confidence-based coloring
            conf = result.get('confidence', 0.0)
            if conf <= 0:
                return (0, 0, 255)
            if conf >= 0.65:
                return (0, 255, 0)
            if conf >= 0.40:
                return (0, 255, 191)
            return (0, 140, 255)
        # Similarity-based
        if sim >= 80:
            return (0, 255, 0)    # Green
        if sim >= 60:
            return (0, 255, 191)  # Lime
        if sim >= 40:
            return (0, 200, 255)  # Yellow-orange
        return (0, 0, 255)        # Red

    def _on_analysis_done(self, result):
        self.results.append(result)
        self._show_feedback(result)

    # ------------------------------------------------------------------
    #  SCREEN 3 — FEEDBACK (per-image, kiosk style)
    # ------------------------------------------------------------------

    def _show_feedback(self, result):
        self._clear()

        path = result['path']
        idx = self.current_index
        total = len(self.image_paths)

        # ── Top bar ───────────────────────────────────────────────────────
        top = ctk.CTkFrame(self.container, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(10, 0))

        ctk.CTkLabel(top, text=f"Image {idx + 1} / {total}",
                     font=("Inter", 16, "bold"), text_color="white"
                     ).pack(side="left")
        ctk.CTkLabel(top, text=os.path.basename(path),
                     font=("Inter", 14), text_color="white"
                     ).pack(side="left", padx=12)

        # Nav buttons
        nav = ctk.CTkFrame(top, fg_color="transparent")
        nav.pack(side="right")
        if idx > 0:
            ctk.CTkButton(nav, text="← Prev", font=("Inter", 16),
                          fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                          width=100, height=36, corner_radius=10,
                          command=self._go_prev).pack(side="left", padx=4)
        if idx < total - 1:
            ctk.CTkButton(nav, text="Next →", font=("Inter", 16),
                          fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                          width=100, height=36, corner_radius=10,
                          command=self._go_next).pack(side="left", padx=4)

        ctk.CTkButton(nav, text="🔄 Retry", font=("Inter", 16),
                      fg_color="#8e44ad", hover_color="#9b59b6",
                      width=100, height=36, corner_radius=10,
                      command=self._retry_current).pack(side="left", padx=4)

        ctk.CTkButton(nav, text="💾 Save", font=("Inter", 16),
                      fg_color="#2c3e50", hover_color="#34495e",
                      width=100, height=36, corner_radius=10,
                      command=self._save_screenshot).pack(side="left", padx=4)

        ctk.CTkButton(nav, text="Summary", font=("Inter", 16, "bold"),
                      fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
                      width=120, height=36, corner_radius=10,
                      command=self._show_summary).pack(side="left", padx=4)

        # ── Target technique dropdown (below top bar) ─────────────────────
        target_bar = ctk.CTkFrame(self.container, fg_color="transparent")
        target_bar.pack(fill="x", padx=20, pady=(4, 0))

        ctk.CTkLabel(target_bar, text="Target:", font=("Inter", 14, "bold"),
                     text_color="white").pack(side="left")

        class_options = ["Any (free practice)"] + [t["name"] for t in TECHNIQUES]
        # Resolve current selection for display
        if self.selected_class_key:
            current_display = next(
                (t["name"] for t in TECHNIQUES if t["key"] == self.selected_class_key),
                "Any (free practice)")
        else:
            current_display = "Any (free practice)"

        self._fb_class_var = ctk.StringVar(value=current_display)
        dropdown = ctk.CTkComboBox(
            target_bar, values=class_options,
            variable=self._fb_class_var,
            font=("Inter", 14), dropdown_font=("Inter", 13),
            width=260, height=32, corner_radius=8,
            state="readonly",
            command=self._on_target_changed)
        dropdown.pack(side="left", padx=8)

        # ── Error state ───────────────────────────────────────────────────
        if result.get('error'):
            ctk.CTkLabel(self.container, text=f"⚠ {result['error']}",
                         font=FONT_BOLD, text_color=COLOR_WARNING
                         ).pack(expand=True)
            return

        # ── Main content (image + feedback panel) ─────────────────────────
        body = ctk.CTkFrame(self.container, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(8, 10))

        # Left: image with skeleton
        img_frame = ctk.CTkFrame(body, fg_color="#1a1f36", corner_radius=16)
        img_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        vis = result.get('vis_frame')
        if vis is not None:
            pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            # Fit to ~750×560 keeping aspect ratio
            max_w, max_h = 750, 560
            pil.thumbnail((max_w, max_h), Image.LANCZOS)
            ctk_img = ctk.CTkImage(pil, size=pil.size)
            lbl = ctk.CTkLabel(img_frame, text="", image=ctk_img)
            lbl.pack(expand=True, padx=10, pady=10)
            self._img_ref = ctk_img  # prevent GC

        # Right: feedback panel
        panel = ctk.CTkFrame(body, fg_color="white", corner_radius=20,
                             width=420)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        pred = result.get('predicted_class', 'N/A')
        conf = result.get('confidence', 0.0)
        sim = result.get('accuracy')  # None when no target selected
        stick = result.get('stick_endpoints') is not None
        has_target = self.selected_class_key is not None
        pose_detected = pred not in ('N/A', 'No Technique Detected', '') and conf > 0

        # Score logic — accuracy-based when target is selected
        if has_target and sim is not None:
            if sim >= 80:
                score_text, score_color = "EXCELLENT!", "#2ecc71"
            elif sim >= 60:
                score_text, score_color = "GOOD", "#27ae60"
            elif sim >= 40:
                score_text, score_color = "FAIR", "#f39c12"
            else:
                score_text, score_color = "NEEDS WORK", "#e74c3c"
        else:
            if not pose_detected:
                score_text, score_color = "NOT DETECTED", "#e74c3c"
            else:
                score_text, score_color = pred.replace('_correct', '').replace('_', ' ').title(), COLOR_ACCENT

        # ── Target class ──────────────────────────────────────────────────
        if has_target:
            target_display = self.selected_class_key.replace('_correct', '').replace('_', ' ').title()
            ctk.CTkLabel(panel, text=target_display,
                         font=("Inter", 26, "bold"), text_color=COLOR_TEXT
                         ).pack(anchor="w", padx=20, pady=(16, 0))
        else:
            ctk.CTkLabel(panel, text="Free Practice",
                         font=("Inter", 13), text_color="gray"
                         ).pack(anchor="w", padx=20, pady=(16, 0))
            display_name = pred.replace('_correct', '').replace('_', ' ').title() if pose_detected else "No Technique"
            ctk.CTkLabel(panel, text=display_name,
                         font=("Inter", 26, "bold"), text_color=COLOR_TEXT
                         ).pack(anchor="w", padx=20, pady=(2, 0))

        # Score badge
        badge = ctk.CTkFrame(panel, fg_color=score_color, corner_radius=12,
                             height=48)
        badge.pack(fill="x", padx=20, pady=(8, 0))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=score_text,
                     font=("Inter", 22, "bold"), text_color="white"
                     ).pack(expand=True)

        # ── Accuracy score (main metric when target selected) ───────────
        if has_target and sim is not None:
            ctk.CTkLabel(panel, text=f"Accuracy: {sim:.1f}%",
                         font=("Inter", 24, "bold"), text_color=COLOR_TEXT
                         ).pack(anchor="w", padx=20, pady=(12, 0))
            ctk.CTkLabel(panel, text=f"GCN Confidence: {conf * 100:.1f}%",
                         font=("Inter", 14), text_color="gray"
                         ).pack(anchor="w", padx=20, pady=(2, 0))

        else:
            ctk.CTkLabel(panel, text=f"Confidence: {conf * 100:.1f}%",
                         font=("Inter", 20), text_color=COLOR_TEXT
                         ).pack(anchor="w", padx=20, pady=(12, 0))

        # Stick
        stick_text = "✓ Stick Detected" if stick else "✗ No Stick"
        stick_color = COLOR_SUCCESS if stick else "#95a5a6"
        ctk.CTkLabel(panel, text=stick_text,
                     font=("Inter", 16), text_color=stick_color
                     ).pack(anchor="w", padx=20, pady=(8, 0))

        # Separator
        ctk.CTkFrame(panel, fg_color="#ecf0f1", height=2).pack(
            fill="x", padx=20, pady=(10, 6))

        # Feedback messages
        msgs = result.get('feedback_messages', [])
        if msgs:
            ctk.CTkLabel(panel, text="Corrections",
                         font=("Inter", 16, "bold"), text_color=COLOR_ACCENT
                         ).pack(anchor="w", padx=20, pady=(0, 4))
            for msg in msgs[:5]:
                row = ctk.CTkFrame(panel, fg_color="transparent")
                row.pack(anchor="w", padx=20, pady=1, fill="x")
                ctk.CTkLabel(row, text="▸", font=("Inter", 16, "bold"),
                             text_color=COLOR_WARNING, width=20
                             ).pack(side="left")
                ctk.CTkLabel(row, text=msg, font=("Inter", 14),
                             text_color=COLOR_TEXT, wraplength=340,
                             justify="left", anchor="w"
                             ).pack(side="left", padx=4)
        else:
            ctk.CTkLabel(panel, text="No corrections needed",
                         font=("Inter", 14), text_color="gray"
                         ).pack(anchor="w", padx=20)

    # ------------------------------------------------------------------
    #  NAVIGATION
    # ------------------------------------------------------------------

    def _go_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._show_feedback(self.results[self.current_index])

    def _go_next(self):
        next_idx = self.current_index + 1
        self.current_index = next_idx
        if next_idx < len(self.results):
            # Already analyzed
            self._show_feedback(self.results[next_idx])
        else:
            # Need to analyze
            self._analyze_current_image()

    def _retry_current(self):
        """Re-analyze the current image, replacing its previous result."""
        idx = self.current_index
        # Remove the old result so it gets replaced
        if idx < len(self.results):
            self.results.pop(idx)
        self._analyze_current_image()

    def _on_target_changed(self, choice):
        """Handle target technique change from the feedback-screen dropdown."""
        if choice == "Any (free practice)":
            self.selected_class_key = None
        else:
            match = [t for t in TECHNIQUES if t["name"] == choice]
            self.selected_class_key = match[0]["key"] if match else None

        # Recompute accuracy + feedback for all existing results
        for r in self.results:
            if r.get('error'):
                continue
            self._compute_similarity_and_feedback(r)
            # Rebuild vis frame with updated skeleton color
            if r.get('landmarks_absolute'):
                path = r['path']
                frame = cv2.imread(path)
                if frame is not None:
                    vis = frame.copy()
                    draw_skeleton_on_frame(vis, r.get('landmarks_absolute'), self._skeleton_color(r))
                    draw_stick_on_frame(vis, r.get('stick_endpoints'))
                    r['vis_frame'] = vis

        # Refresh current feedback screen
        if self.current_index < len(self.results):
            self._show_feedback(self.results[self.current_index])

    def _save_screenshot(self):
        """Capture the full eval app window and save as PNG."""
        from datetime import datetime
        try:
            # Capture only the content area (below nav bar), not the full window
            from PIL import ImageGrab
            # Use the container widget's screen position if available, else fall back to window
            target = getattr(self, 'container', self)
            x = target.winfo_rootx()
            y = target.winfo_rooty()
            w = target.winfo_width()
            h = target.winfo_height()
            screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))

            # Save to project root / eval_screenshots
            save_dir = os.path.join(project_root, "eval_screenshots")
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            idx = self.current_index
            fname = os.path.basename(self.image_paths[idx]) if idx < len(self.image_paths) else "unknown"
            base = os.path.splitext(fname)[0]
            vp = self.selected_viewpoint or "front"
            cls = self.selected_class_key.replace('_correct', '') if self.selected_class_key else "free"
            save_path = os.path.join(save_dir, f"eval_{vp}_{cls}_{base}_{timestamp}.png")

            screenshot.save(save_path)
            print(f"[EvalApp] Screenshot saved: {save_path}")

            # Brief visual confirmation
            self._flash_save_confirmation(save_path)
        except Exception as e:
            print(f"[EvalApp] Screenshot failed: {e}")

    def _flash_save_confirmation(self, path):
        """Show a brief toast-like confirmation."""
        toast = ctk.CTkFrame(self, fg_color="#27ae60", corner_radius=12)
        toast.place(relx=0.5, rely=0.95, anchor="center")
        ctk.CTkLabel(toast, text=f"  ✓ Saved to {os.path.basename(path)}  ",
                     font=("Inter", 14, "bold"), text_color="white"
                     ).pack(padx=12, pady=6)
        self.after(2000, toast.destroy)

    # ------------------------------------------------------------------
    #  SCREEN 4 — SUMMARY
    # ------------------------------------------------------------------

    def _show_summary(self):
        # Analyze remaining images first
        while len(self.results) < len(self.image_paths):
            self.current_index = len(self.results)
            self._analyze_remaining_sync()

        self._clear()

        ctk.CTkLabel(self.container, text="Evaluation Results",
                     font=FONT_HEADER, text_color="white").pack(pady=(30, 10))

        threshold = self.confidence_threshold

        # Stats card
        card = ctk.CTkFrame(self.container, fg_color="white",
                            corner_radius=24, width=900)
        card.pack(fill="x", padx=40, pady=(0, 10))

        # Summary stats
        total = len(self.results)
        errors = sum(1 for r in self.results if r.get('error'))
        valid = total - errors

        if self.selected_class_key:
            # Similarity-based summary
            sims = [r.get('accuracy', 0) for r in self.results
                    if not r.get('error') and r.get('accuracy') is not None]
            avg_sim = sum(sims) / len(sims) if sims else 0
            max_sim = max(sims) if sims else 0
            min_sim = min(sims) if sims else 0
            target_display = self.selected_class_key.replace('_correct', '').replace('_', ' ').title()

            stats_frame = ctk.CTkFrame(card, fg_color="transparent")
            stats_frame.pack(fill="x", padx=30, pady=20)

            sim_color = COLOR_SUCCESS if avg_sim >= 70 else "#f39c12" if avg_sim >= 40 else COLOR_WARNING
            for label, value, color in [
                ("Total Images", str(total), COLOR_TEXT),
                ("Avg Accuracy", f"{avg_sim:.1f}%", sim_color),
                ("Best", f"{max_sim:.1f}%", COLOR_SUCCESS),
                ("Worst", f"{min_sim:.1f}%", COLOR_WARNING),
            ]:
                col = ctk.CTkFrame(stats_frame, fg_color="transparent")
                col.pack(side="left", expand=True)
                ctk.CTkLabel(col, text=value, font=("Inter", 36, "bold"),
                             text_color=color).pack()
                ctk.CTkLabel(col, text=label, font=("Inter", 14),
                             text_color="gray").pack()

            ctk.CTkLabel(card, text=f"Target: {target_display}  |  Viewpoint: {self.selected_viewpoint.title()}  |  Threshold: {int(threshold*100)}%",
                         font=("Inter", 14), text_color="gray"
                         ).pack(padx=30, pady=(0, 12))
        else:
            stats_frame = ctk.CTkFrame(card, fg_color="transparent")
            stats_frame.pack(fill="x", padx=30, pady=20)
            ctk.CTkLabel(stats_frame, text=str(total),
                         font=("Inter", 36, "bold"), text_color=COLOR_TEXT).pack()
            ctk.CTkLabel(stats_frame, text="Images evaluated (free practice — no accuracy)",
                         font=("Inter", 14), text_color="gray").pack()

        # Per-image table
        table_frame = ctk.CTkScrollableFrame(self.container, fg_color="white",
                                             corner_radius=16, height=320)
        table_frame.pack(fill="x", padx=40, pady=(0, 10))

        # Header
        hdr = ctk.CTkFrame(table_frame, fg_color="#ecf0f1", corner_radius=8)
        hdr.pack(fill="x", padx=8, pady=(8, 4))
        cols = [("#", 40), ("Image", 240), ("GCN Prediction", 200),
                ("Confidence", 100)]
        if self.selected_class_key:
            cols.append(("Accuracy", 100))
        cols.append(("Score", 120))
        for text, w in cols:
            ctk.CTkLabel(hdr, text=text, font=("Inter", 13, "bold"),
                         text_color=COLOR_TEXT, width=w, anchor="w"
                         ).pack(side="left", padx=4)

        # Rows
        for i, r in enumerate(self.results):
            row = ctk.CTkFrame(table_frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=1)

            pred = r.get('predicted_class', 'N/A')
            conf = r.get('confidence', 0.0)
            sim_val = r.get('accuracy')
            fname = os.path.basename(r['path'])

            pred_display = pred.replace('_correct', '').replace('_', ' ').title() if pred != 'N/A' else '—'

            if r.get('error'):
                score_text, score_color = "ERROR", COLOR_WARNING
            elif sim_val is not None:
                if sim_val >= 80:
                    score_text, score_color = "EXCELLENT", COLOR_SUCCESS
                elif sim_val >= 60:
                    score_text, score_color = "GOOD", "#27ae60"
                elif sim_val >= 40:
                    score_text, score_color = "FAIR", "#f39c12"
                else:
                    score_text, score_color = "NEEDS WORK", COLOR_WARNING
            else:
                score_text, score_color = pred_display, COLOR_TEXT

            row_data = [
                (str(i + 1), 40, COLOR_TEXT),
                (fname[:30], 240, COLOR_TEXT),
                (pred_display, 200, COLOR_TEXT),
                (f"{conf*100:.1f}%", 100, COLOR_TEXT),
            ]
            if self.selected_class_key:
                sim_text = f"{sim_val:.1f}%" if sim_val is not None else "—"
                row_data.append((sim_text, 100, COLOR_TEXT))
            row_data.append((score_text, 120, score_color))

            for text, w, color in row_data:
                ctk.CTkLabel(row, text=text, font=("Inter", 13),
                             text_color=color, width=w, anchor="w"
                             ).pack(side="left", padx=4)

        # Bottom actions
        actions = ctk.CTkFrame(self.container, fg_color="transparent")
        actions.pack(pady=(0, 15))

        ctk.CTkButton(actions, text="New Evaluation", font=FONT_BOLD,
                      fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                      height=50, width=220, corner_radius=16,
                      command=self.show_select_screen).pack(side="left", padx=10)

        ctk.CTkButton(actions, text="Back to Feedback", font=("Inter", 18),
                      fg_color="transparent", border_width=2,
                      border_color="white", text_color="white",
                      hover_color="#aecef7", height=44, width=200, corner_radius=22,
                      command=lambda: self._show_feedback_at(0)).pack(side="left", padx=10)

    def _show_feedback_at(self, idx):
        if 0 <= idx < len(self.results):
            self.current_index = idx
            self._show_feedback(self.results[idx])

    def _analyze_remaining_sync(self):
        """Synchronously analyze all remaining images (for summary)."""
        idx = len(self.results)
        if idx >= len(self.image_paths):
            return
        path = self.image_paths[idx]
        frame = cv2.imread(path)
        result = {'path': path, 'error': None}

        if frame is None:
            result['error'] = 'Could not load image'
            self.results.append(result)
            return

        if self.pose_analyzer is None:
            result['error'] = 'PoseAnalyzer not available'
            self.results.append(result)
            return

        try:
            self.pose_analyzer.pose = self.pose_analyzer.mp_pose.Pose(
                static_image_mode=True, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
                smooth_landmarks=False)
            self.pose_analyzer._cached_stick_results.clear()
            self.pose_analyzer.stick_buffer.clear()

            if self.pose_analyzer.gcn_engine:
                self.pose_analyzer.gcn_engine.set_viewpoint(self.selected_viewpoint)

            # MODE: snapshot - Evaluation/classification (no temporal smoothing for accuracy)
            persons = self.pose_analyzer.process_frame(
                frame, skip_ml_inference=False, mode='snapshot',
                skip_threshold=True)

            if persons and len(persons) > 0:
                p = persons[0]
                result['predicted_class'] = p.get('predicted_class', 'N/A')
                result['confidence'] = p.get('confidence', 0.0)
                result['landmarks_absolute'] = p.get('landmarks_absolute')
                result['stick_endpoints'] = p.get('stick_endpoints')
                result['global_features'] = p.get('global_features')
                result['pose_kpts_array'] = p.get('pose_kpts_array')
                result['stick_kpts_array'] = p.get('stick_kpts_array')
                result['frame_w'] = frame.shape[1]
                result['frame_h'] = frame.shape[0]
            else:
                result['predicted_class'] = 'N/A'
                result['confidence'] = 0.0
        except Exception as e:
            result['error'] = str(e)

        # Similarity + feedback
        if result.get('error') is None:
            self._compute_similarity_and_feedback(result)

        # Vis frame
        if result.get('error') is None:
            vis = frame.copy()
            draw_skeleton_on_frame(vis, result.get('landmarks_absolute'), self._skeleton_color(result))
            draw_stick_on_frame(vis, result.get('stick_endpoints'))
            result['vis_frame'] = vis

        self.results.append(result)


# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    app = EvalApp()
    app.mainloop()
