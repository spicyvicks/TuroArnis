import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
import time
import threading
import numpy as np
import sys
import os
import json

# Add project root to path
# Add project root to path
# In PyInstaller, the app is running from a temp dir, so we need to ensure local imports work
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    base_path = sys._MEIPASS
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    # Also add the app directory specifically if needed for direct imports
    app_path = os.path.join(base_path, 'app')
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
else:
    # Running as script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import App Components
from app.database.db_manager import DatabaseManager
from app.gui.results_window import ResultsWindow
from app.gui.user_dialog import UserManagementDialog
from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.utils.resource_path import get_resource_path

# Fix for CTk DPI Scaling
try:
    from customtkinter.windows.widgets.scaling import ScalingTracker
    ScalingTracker.deactivate_automatic_dpi_awareness = True
except ImportError:
    pass

# Theme & Configuration
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Professional Palette (Light Theme as requested)
COLOR_BG = "#74b9ff"           # Light Blue (Splash Style)
COLOR_ACCENT = "#2980b9"       # Strong Blue for buttons
COLOR_ACCENT_HOVER = "#3498db" # Lighter Blue for hover
COLOR_SUCCESS = "#27ae60"      # Green
COLOR_WARNING = "#e74c3c"      # Red
COLOR_TEXT = "#2c3e50"         # Dark Text for Light BG
COLOR_TEXT_WHITE = "white"
COLOR_PANEL = "white"          # White panels on light blue
FONT_MAIN = ("Inter", 24)
FONT_HEADER = ("Inter", 48, "bold")
FONT_BOLD = ("Inter", 24, "bold")

# Confidence thresholds are defined per-viewpoint in app/models/gcn_model_config.json
# (single source of truth — do not duplicate here)

# ============================================
# TEMPORARY: Demo Mode - Controlled Results
# Set DEMO_MODE = False to use normal camera
# ============================================
DEMO_MODE = False
DEMO_IMAGE_RESULTS = {
    "excellent.jpg": {
        "confidence": 0.85, 
        "pose": "left_elbow_block_correct",
        "force_feedback": ["Maintain position"],
        "force_excellent": True
    },
    "good.jpg": {
        "confidence": 0.40, 
        "pose": "left_elbow_block_correct",
        "force_feedback": None,
        "force_excellent": False
    },
    "fail.jpg": {
        "confidence": 0.00, 
        "pose": "left_elbow_block_correct",
        "force_feedback": ["Get into Left Elbow Block position", "Face the camera fully"],
        "force_excellent": False
    },
}
DEMO_IMAGE_PATH = "demo_images/good.jpg"  # Change for each test: excellent.jpg, good.jpg, fail.jpg
# ============================================

class AppState:
    SPLASH             = "splash"
    MODE_SELECT        = "mode_select"
    USER_COUNT         = "user_count"
    CONFIG             = "config"
    ZONING             = "zoning"
    COUNTDOWN          = "countdown"
    SNAPSHOT           = "snapshot"
    FEEDBACK           = "feedback"
    PAUSED             = "paused"
    RESULTS            = "results"
    LESSON_SELECT      = "lesson_select"
    LESSON_INSTRUCTION = "lesson_instruction"

# --- LESSON TECHNIQUE CATALOGUE ---
# Keys match gcn_model_config.json class_names
TECHNIQUES = [
    {"key": "crown_thrust_correct",       "name": "Crown Thrust",       "category": "Thrust",
     "description": "A straight thrust aimed at the crown of the head.",
     "key_points": ["Extend your striking arm fully forward",
                    "Align the stick tip with the opponent's crown",
                    "Drive forward from the shoulder, not just the wrist",
                    "Keep your non-striking arm guarded at the chest"],
     "viewpoint": "Front"},
    {"key": "left_chest_thrust_correct",   "name": "Left Chest Thrust",  "category": "Thrust",
     "description": "A thrust directed at the left side of the opponent's chest.",
     "key_points": ["Rotate torso slightly left to load the strike",
                    "Extend arm to full reach at chest height",
                    "Step forward on the dominant foot for power",
                    "Keep the tip pointed directly at the target"],
     "viewpoint": "Front"},
    {"key": "left_elbow_block_correct",    "name": "Left Elbow Block",   "category": "Block",
     "description": "A downward-angled block protecting the left elbow area.",
     "key_points": ["Bring your stick diagonally across the body to the left",
                    "Elbow stays close to the rib cage",
                    "Absorb the force through your forearm, not the wrist",
                    "Maintain a slight bend in the blocking arm"],
     "viewpoint": "Front"},
    {"key": "left_eye_thrust_correct",     "name": "Left Eye Thrust",    "category": "Thrust",
     "description": "A precise thrust targeting the left eye / temple area.",
     "key_points": ["Raise stick to eye level before thrusting",
                    "Extend arm in a straight line — no arc",
                    "Feet shoulder-width apart for balance",
                    "Non-striking hand covers your own face"],
     "viewpoint": "Front"},
    {"key": "left_knee_block_correct",     "name": "Left Knee Block",    "category": "Block",
     "description": "A low block defending the left knee from downward strikes.",
     "key_points": ["Drop the stick tip low toward the left knee",
                    "Slightly bend both knees for a stable base",
                    "Keep the back straight — do not hunch",
                    "Redirect rather than absorb the incoming strike"],
     "viewpoint": "Front"},
    {"key": "left_temple_block_correct",   "name": "Left Temple Block",  "category": "Block",
     "description": "A high block protecting the left temple.",
     "key_points": ["Raise the stick above shoulder height on the left side",
                    "Angle the stick at roughly 45\u00b0 outward",
                    "Keep your elbow slightly bent to absorb impact",
                    "Eyes forward — don't look at the stick"],
     "viewpoint": "Front"},
    {"key": "right_chest_thrust_correct",  "name": "Right Chest Thrust", "category": "Thrust",
     "description": "A thrust directed at the right side of the opponent's chest.",
     "key_points": ["Rotate torso slightly right to load the strike",
                    "Extend arm to full reach at chest height",
                    "Step forward on the dominant foot for power",
                    "Keep the tip pointed directly at the target"],
     "viewpoint": "Front"},
    {"key": "right_elbow_block_correct",   "name": "Right Elbow Block",  "category": "Block",
     "description": "A downward-angled block protecting the right elbow area.",
     "key_points": ["Bring your stick diagonally across the body to the right",
                    "Elbow stays close to the rib cage",
                    "Absorb the force through your forearm, not the wrist",
                    "Maintain a slight bend in the blocking arm"],
     "viewpoint": "Front"},
    {"key": "right_eye_thrust_correct",    "name": "Right Eye Thrust",   "category": "Thrust",
     "description": "A precise thrust targeting the right eye / temple area.",
     "key_points": ["Raise stick to eye level before thrusting",
                    "Extend arm in a straight line — no arc",
                    "Feet shoulder-width apart for balance",
                    "Non-striking hand covers your own face"],
     "viewpoint": "Front"},
    {"key": "right_knee_block_correct",    "name": "Right Knee Block",   "category": "Block",
     "description": "A low block defending the right knee from downward strikes.",
     "key_points": ["Drop the stick tip low toward the right knee",
                    "Slightly bend both knees for a stable base",
                    "Keep the back straight — do not hunch",
                    "Redirect rather than absorb the incoming strike"],
     "viewpoint": "Front"},
    {"key": "right_temple_block_correct",  "name": "Right Temple Block", "category": "Block",
     "description": "A high block protecting the right temple.",
     "key_points": ["Raise the stick above shoulder height on the right side",
                    "Angle the stick at roughly 45\u00b0 outward",
                    "Keep your elbow slightly bent to absorb impact",
                    "Eyes forward — don't look at the stick"],
     "viewpoint": "Front"},
    {"key": "solar_plexus_thrust_correct", "name": "Solar Plexus Thrust", "category": "Thrust",
     "description": "A mid-body thrust aimed at the solar plexus.",
     "key_points": ["Target the center of the torso at stomach height",
                    "Lead with the hip to generate forward momentum",
                    "Arm fully extended at the point of contact",
                    "Keep shoulders level throughout the motion"],
     "viewpoint": "Front"},
]

CATEGORY_COLORS = {
    "Thrust": "#6c3483",   # Dark Purple
    "Block":  "#1a5276",   # Dark Navy Blue
}

# Maps each technique key to its lesson GIF filename (in lesson/{vp}_gif/)
LESSON_IMAGE_MAP = {
    "crown_thrust_correct":       "crown.gif",
    "left_chest_thrust_correct":  "left_chest.gif",
    "left_elbow_block_correct":   "left_elbow.gif",
    "left_eye_thrust_correct":    "left_eye.gif",
    "left_knee_block_correct":    "left_knee.gif",
    "left_temple_block_correct":  "left_temple.gif",
    "right_chest_thrust_correct": "right_chest.gif",
    "right_elbow_block_correct":  "right_elbow.gif",
    "right_eye_thrust_correct":   "right_eye.gif",
    "right_knee_block_correct":   "right_knee.gif",
    "right_temple_block_correct": "right_temple.gif",
    "solar_plexus_thrust_correct":"solar_plexus.gif",
}

def _lesson_gif_filename(technique_key: str, viewpoint: str) -> str:
    """Return the correct GIF filename for a technique + viewpoint combo."""
    return LESSON_IMAGE_MAP.get(technique_key, "")

MAX_GIF_FRAMES = 30  # cap to prevent memory exhaustion on large GIFs

def _load_gif_frames(gif_path: str, max_size: tuple) -> list:
    """Load frames from an animated GIF, sampling evenly if too many frames.
    Returns list of (CTkImage, delay_ms).  Capped at MAX_GIF_FRAMES."""
    frames = []
    try:
        pil_img = Image.open(gif_path)
        n_frames = getattr(pil_img, 'n_frames', 1)

        # Determine which frame indices to keep
        if n_frames <= MAX_GIF_FRAMES:
            indices = list(range(n_frames))
        else:
            # Sample evenly across the animation
            indices = [int(i * n_frames / MAX_GIF_FRAMES) for i in range(MAX_GIF_FRAMES)]

        for i in indices:
            pil_img.seek(i)
            frame = pil_img.copy().convert("RGBA")
            frame.thumbnail(max_size, Image.LANCZOS)
            delay = pil_img.info.get('duration', 100)  # ms per frame
            if delay < 20:
                delay = 100  # safety floor
            ctk_img = ctk.CTkImage(frame, size=frame.size)
            frames.append((ctk_img, delay))
    except Exception as e:
        print(f"[LESSON] Error loading GIF frames from {gif_path}: {e}")
    return frames

class KioskApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("TuroArnis Kiosk")
        self.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        self.attributes("-fullscreen", True)
        
        # Database
        self.db = DatabaseManager('turoarnis.db')
        
        # Initialize Pose Analyzer with GCN
        try:
            print("[Kiosk] Initializing Pose Analyzer with GCN...")
            stick_model_path = get_resource_path('deployment_package/weights/best.pt')
            self.pose_analyzer = PoseAnalyzer(
                detection_interval=3,
                stick_model_path=stick_model_path,
                debug_stick=False
            )
            print("[Kiosk] Pose Analyzer initialized successfully")
        except Exception as e:
            print(f"[Kiosk] Warning: Could not initialize Pose Analyzer: {e}")
            self.pose_analyzer = None

        # Initialize Feedback Analyzer
        try:
            self.feedback_analyzer = FeedbackAnalyzer()
            print("[Kiosk] Feedback Analyzer initialized successfully")
        except Exception as e:
            print(f"[Kiosk] Warning: Could not initialize Feedback Analyzer: {e}")
            self.feedback_analyzer = None
        
        # Bindings
        self.bind("<Escape>", lambda e: self.close_app())
        self.bind("<space>", self.on_spacebar)
        self.bind("<Return>", self.on_enter)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # State
        self.app_state = AppState.SPLASH
        self.previous_state = None
        self.num_users = 0
        self.user_configs = [] 
        self.feedback_timer = 0
        self.countdown_timer = 0
        self.analysis_results = {}  # Store pose analysis per user zone
        self.current_lesson = None  # Set when in guided lesson mode
        
        # Main Container
        self.container = ctk.CTkFrame(self, fg_color="black")
        self.container.pack(fill="both", expand=True)
        
        # Camera Feed
        self.cap = self.get_available_camera()
        self.camera_active = (self.cap is not None)
        self.current_frame = None
        self.frozen_frame = None
        self.running = True
        
        # Video Canvas
        self.video_canvas = tk.Canvas(self.container, bg=COLOR_BG, highlightthickness=0)
        self.video_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        
        # Canvas Items Tracker
        self.canvas_items = []
        self.image_item = None
        self.count_text_id = None
        self.countdown_circle_id = None
        self.timer_text_id = None
        self.show_user_names = True
        self.names_shown_time = 0
        self.realtime_pose_cache = {} # Cache for sharing pose data between threads/loops
        
        # Video display rect (updated each frame for letterbox-aware positioning)
        self.video_x_offset = 0
        self.video_y_offset = 0
        self.video_display_width = self.screen_width
        self.video_display_height = self.screen_height
        
        # Configuration for feedback
        self.feature_templates = {}
        try:
            template_path = get_resource_path('app/models/gcn/feature_templates.json')
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    self.feature_templates = json.load(f)
                print("[Kiosk] Loaded feature templates for feedback")
        except Exception as e:
            print(f"[Kiosk] Warning: Could not load feature templates: {e}")

        # Start Loop
        self.frame_counter = 0
        self.after(100, self.update_feed)
        self.show_splash()
    
    def get_available_camera(self):
        if DEMO_MODE:
            print("[DEMO] Demo mode - no camera needed")
            return None
        print("Searching for cameras...")
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
                if not cap.isOpened():
                    cap = cv2.VideoCapture(i) 
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"SUCCESS: Camera found at index {i}")
                        return cap
                    else:
                        cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {e}")
        return None
    
    def close_app(self):
        self.running = False
        if self.cap: self.cap.release()
        self.db.close()
        self.destroy()
        sys.exit(0)

    def clear_ui(self):
        # Cancel any running GIF animations
        for attr in ('_lesson_gif_anim_id', '_lesson_zoom_anim_id'):
            anim_id = getattr(self, attr, None)
            if anim_id is not None:
                self.after_cancel(anim_id)
                setattr(self, attr, None)
        for item in self.canvas_items:
            self.video_canvas.delete(item)
        self.canvas_items = []

    def add_widget(self, x, y, widget, anchor="center"):
        item = self.video_canvas.create_window(x, y, window=widget, anchor=anchor)
        self.canvas_items.append(item)
        return item

    def add_text(self, x, y, text, font=FONT_MAIN, fill=COLOR_TEXT, anchor="center", no_shadow=False):
        # Shadow
        if not no_shadow and (fill == "white" or fill == COLOR_TEXT_WHITE):
             shadow = self.video_canvas.create_text(x+2, y+2, text=text, font=font, fill="black", anchor=anchor)
             self.canvas_items.append(shadow)
        main = self.video_canvas.create_text(x, y, text=text, font=font, fill=fill, anchor=anchor)
        self.canvas_items.append(main)
        return main

    # --- INPUT HANDLERS ---
    def on_spacebar(self, event):
        if self.app_state == AppState.PAUSED:
            self.resume_session()
        elif self.app_state in [AppState.ZONING, AppState.COUNTDOWN, AppState.SNAPSHOT, AppState.FEEDBACK]:
            self.show_pause_menu()
            
    def on_enter(self, event):
        if self.app_state == AppState.CONFIG:
            self.start_zoning_check()

    # --- MENU ACTIONS ---
    def show_pause_menu(self):
        self.previous_state = self.app_state
        self.app_state = AppState.PAUSED
        self.clear_ui()
        
        cx, cy = self.screen_width // 2, self.screen_height // 2
        
        self.add_text(cx, cy - 150, "SESSION PAUSED", font=("Inter", 64, "bold"), fill="white")
        
        btn_resume = ctk.CTkButton(self.video_canvas, text="RESUME", font=FONT_BOLD,
                                  fg_color=COLOR_SUCCESS, width=300, height=70, corner_radius=15,
                                  command=self.resume_session)
        self.add_widget(cx, cy - 30, btn_resume)

        btn_settings = ctk.CTkButton(self.video_canvas, text="RESTART SETUP", font=FONT_BOLD,
                                    fg_color="#f39c12", width=300, height=70, corner_radius=15,
                                    command=self.show_mode_select)
        self.add_widget(cx, cy + 60, btn_settings)

        btn_quit = ctk.CTkButton(self.video_canvas, text="END SESSION", font=FONT_BOLD,
                                fg_color=COLOR_WARNING, width=300, height=70, corner_radius=15,
                                command=self.end_session_and_show_results)
        self.add_widget(cx, cy + 150, btn_quit)

    def resume_session(self):
        if self.previous_state:
            self.app_state = self.previous_state
            if self.app_state == AppState.ZONING: self.start_zoning_check()
            elif self.app_state == AppState.FEEDBACK: self.show_feedback() 
            elif self.app_state == AppState.COUNTDOWN: self.start_countdown()

    def end_session_and_show_results(self):
        for config in self.user_configs:
            if config.get('session_id'):
                self.db.end_session(config['session_id'])
                config['session_id'] = None
        self.show_results_screen()

    # --- STATES ---

    def show_splash(self):
        self.app_state = AppState.SPLASH
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG) 
        
        cx, cy = self.screen_width // 2, self.screen_height // 2
        
        # Add logo
        try:
            logo_path = get_resource_path('app/assets/TA.png')
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((150, 150), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_img)
                logo_item = self.video_canvas.create_image(cx, cy - 180, image=logo_photo)
                self.canvas_items.append(logo_item)
                self.video_canvas.logo_photo = logo_photo  # Keep reference
        except Exception as e:
            print(f"[Splash] Could not load logo: {e}")
        
        self.add_text(cx, cy - 40, "TuroArnis", font=("Inter", 96, "bold"), fill="white", no_shadow=True)
        self.add_text(cx, cy + 60, "Arnis Form Correction", font=("Inter", 32), fill="white", no_shadow=True)
        
        btn = ctk.CTkButton(self.video_canvas, text="START PRACTICE", font=("Inter", 28, "bold"),
                           fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, height=80, width=300, corner_radius=40,
                           command=self.show_mode_select)
        self.add_widget(cx, cy + 180, btn)

    # ── MODE SELECT ───────────────────────────────────────────────────────────

    def show_mode_select(self):
        self.app_state = AppState.MODE_SELECT
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)
        cx, cy = self.screen_width // 2, self.screen_height // 2

        self.add_text(cx, cy - 220, "Choose Your Mode", font=FONT_HEADER, fill="white")

        # Free Practice card
        fp_card = ctk.CTkFrame(self.video_canvas, fg_color="white",
                               corner_radius=24, width=380, height=300)
        fp_card.pack_propagate(False)
        ctk.CTkLabel(fp_card, text="🥋", font=("Inter", 56)).pack(pady=(28, 4))
        ctk.CTkLabel(fp_card, text="Free Practice",
                     font=("Inter", 26, "bold"), text_color=COLOR_TEXT).pack()
        ctk.CTkLabel(fp_card, text="Strike any technique.\nThe system will recognise it.",
                     font=("Inter", 15), text_color="gray", justify="center").pack(pady=8)
        ctk.CTkButton(fp_card, text="SELECT", font=FONT_BOLD,
                      fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                      height=48, corner_radius=12,
                      command=self._enter_free_practice).pack(padx=30, fill="x", pady=(0, 20))
        self.add_widget(cx - 240, cy + 20, fp_card)

        # Guided Lesson card
        gl_card = ctk.CTkFrame(self.video_canvas, fg_color="white",
                               corner_radius=24, width=380, height=300)
        gl_card.pack_propagate(False)
        ctk.CTkLabel(gl_card, text="📖", font=("Inter", 56)).pack(pady=(28, 4))
        ctk.CTkLabel(gl_card, text="Guided Lesson",
                     font=("Inter", 26, "bold"), text_color=COLOR_TEXT).pack()
        ctk.CTkLabel(gl_card, text="Pick a technique to learn.\nGet targeted feedback.",
                     font=("Inter", 15), text_color="gray", justify="center").pack(pady=8)
        ctk.CTkButton(gl_card, text="SELECT", font=FONT_BOLD,
                      fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
                      height=48, corner_radius=12,
                      command=self.show_lesson_select).pack(padx=30, fill="x", pady=(0, 20))
        self.add_widget(cx + 240, cy + 20, gl_card)

        btn_back = ctk.CTkButton(self.video_canvas, text="← Back", font=("Inter", 18),
                                 fg_color="transparent", border_width=2,
                                 border_color="white", text_color="white",
                                 hover_color="#aecef7", height=44, width=160, corner_radius=22,
                                 command=self.show_splash)
        self.add_widget(cx, cy + 290, btn_back)

    def _enter_free_practice(self):
        """Free Practice path — resets lesson context and goes to user count."""
        self.current_lesson = None
        self.show_user_count()

    # ── LESSON SELECT ─────────────────────────────────────────────────────────

    def show_lesson_select(self):
        self.app_state = AppState.LESSON_SELECT
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)
        cx = self.screen_width // 2
        sh = self.screen_height

        self.add_text(cx, 80, "Select a Technique", font=("Inter", 48, "bold"), fill="white")
        self.add_text(cx, 130, "Choose one of the 12 fundamental techniques to practise",
                      font=("Inter", 20), fill="white")

        # Back button (top-left)
        btn_back = ctk.CTkButton(self.video_canvas, text="← Back",
                                 font=("Inter", 18),
                                 fg_color="transparent", border_width=2,
                                 border_color="white", text_color="white",
                                 hover_color="#aecef7", height=44, width=160, corner_radius=22,
                                 command=self.show_mode_select)
        self.add_widget(120, 60, btn_back)

        # Scrollable card grid
        wrapper = ctk.CTkFrame(self.video_canvas, fg_color="transparent",
                               width=1120, height=sh - 240)
        self.add_widget(cx, (sh // 2) + 60, wrapper)
        wrapper.pack_propagate(False)

        scroll = ctk.CTkScrollableFrame(wrapper, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        cols = 3
        for idx, tech in enumerate(TECHNIQUES):
            row, col = divmod(idx, cols)
            cat_color = CATEGORY_COLORS.get(tech["category"], COLOR_ACCENT)

            card = ctk.CTkFrame(scroll, fg_color="white", corner_radius=18,
                                width=340, height=170)
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")
            card.pack_propagate(False)
            card.grid_propagate(False)

            badge = ctk.CTkFrame(card, fg_color=cat_color, corner_radius=8, width=70, height=26)
            badge.pack(anchor="w", padx=16, pady=(14, 0))
            badge.pack_propagate(False)
            ctk.CTkLabel(badge, text=tech["category"],
                         font=("Inter", 12, "bold"), text_color="white").pack(expand=True)

            ctk.CTkLabel(card, text=tech["name"],
                         font=("Inter", 20, "bold"), text_color=COLOR_TEXT,
                         anchor="w").pack(anchor="w", padx=16, pady=(4, 0))
            ctk.CTkLabel(card, text=tech["description"],
                         font=("Inter", 14), text_color="gray",
                         wraplength=290, justify="left", anchor="w").pack(anchor="w", padx=16)
            ctk.CTkButton(card, text="Learn →",
                          font=("Inter", 15, "bold"),
                          fg_color=cat_color, hover_color=COLOR_ACCENT,
                          height=34, corner_radius=10,
                          command=lambda t=tech: self.show_lesson_viewpoint_select(t)).pack(
                              anchor="e", padx=16, pady=(6, 12))

    # ── LESSON VIEWPOINT SELECTION ────────────────────────────────────────────

    def show_lesson_viewpoint_select(self, technique: dict):
        """Show viewpoint selection screen before starting lesson practice."""
        self.app_state = AppState.LESSON_SELECT
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)
        
        cx, cy = self.screen_width // 2, self.screen_height // 2
        cat_color = CATEGORY_COLORS.get(technique["category"], COLOR_ACCENT)
        
        # Main container card
        card = ctk.CTkFrame(self.video_canvas, fg_color="white",
                            corner_radius=24, width=600, height=480)
        card.pack_propagate(False)
        self.add_widget(cx, cy, card)
        
        # Category badge
        badge = ctk.CTkFrame(card, fg_color=cat_color, corner_radius=10,
                              width=100, height=32)
        badge.pack(anchor="center", pady=(40, 0))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=technique["category"],
                     font=("Inter", 14, "bold"), text_color="white").pack(expand=True)
        
        # Technique name
        ctk.CTkLabel(card, text=technique["name"],
                     font=("Inter", 32, "bold"), text_color=COLOR_TEXT).pack(pady=(20, 8))
        
        # Instruction text
        ctk.CTkLabel(card, text="Which camera angle will you practice from?",
                     font=("Inter", 18), text_color=COLOR_TEXT).pack(pady=(0, 30))
        
        # Viewpoint buttons container
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        # Store technique for callback
        self._pending_technique = technique
        
        # Front button
        front_btn = ctk.CTkButton(btn_frame, text="Front",
                                  font=("Inter", 18, "bold"),
                                  fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                                  height=70, width=160, corner_radius=15,
                                  command=lambda: self._start_lesson_with_viewpoint("Front"))
        front_btn.pack(side="left", padx=12)
        
        # Left Side button
        left_btn = ctk.CTkButton(btn_frame, text="Left Side",
                                 font=("Inter", 18, "bold"),
                                 fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                                 height=70, width=160, corner_radius=15,
                                 command=lambda: self._start_lesson_with_viewpoint("Left Side"))
        left_btn.pack(side="left", padx=12)
        
        # Right Side button
        right_btn = ctk.CTkButton(btn_frame, text="Right Side",
                                  font=("Inter", 18, "bold"),
                                  fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
                                  height=70, width=160, corner_radius=15,
                                  command=lambda: self._start_lesson_with_viewpoint("Right Side"))
        right_btn.pack(side="left", padx=12)
        
        # Back button
        ctk.CTkButton(card, text="← Back",
                      font=("Inter", 16),
                      fg_color="transparent", border_width=2,
                      border_color=COLOR_TEXT, text_color=COLOR_TEXT,
                      hover_color="#ecf0f1", height=44, width=140, corner_radius=22,
                      command=self.show_lesson_select).pack(pady=(30, 0))
    
    def _start_lesson_with_viewpoint(self, viewpoint: str):
        """Start lesson with selected viewpoint."""
        technique = self._pending_technique.copy()
        technique["viewpoint"] = viewpoint
        self.show_lesson_instruction(technique)

    # ── LESSON INSTRUCTION ────────────────────────────────────────────────────

    def show_lesson_instruction(self, technique: dict):
        self.current_lesson = technique
        self.app_state = AppState.LESSON_INSTRUCTION
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)

        cx, cy = self.screen_width // 2, self.screen_height // 2
        cat_color = CATEGORY_COLORS.get(technique["category"], COLOR_ACCENT)

        # Left panel — technique info
        info_panel = ctk.CTkFrame(self.video_canvas, fg_color="white",
                                  corner_radius=24, width=540, height=620)
        info_panel.pack_propagate(False)
        self.add_widget(cx - 315, cy, info_panel)

        badge = ctk.CTkFrame(info_panel, fg_color=cat_color, corner_radius=10,
                              width=90, height=30)
        badge.pack(anchor="w", padx=24, pady=(24, 0))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=technique["category"],
                     font=("Inter", 13, "bold"), text_color="white").pack(expand=True)

        ctk.CTkLabel(info_panel, text=technique["name"],
                     font=("Inter", 36, "bold"), text_color=COLOR_TEXT,
                     anchor="w").pack(anchor="w", padx=24, pady=(8, 4))
        ctk.CTkLabel(info_panel, text=technique["description"],
                     font=("Inter", 18), text_color="gray",
                     wraplength=480, justify="left", anchor="w").pack(anchor="w", padx=24, pady=(0, 16))

        sep = ctk.CTkFrame(info_panel, fg_color="#ecf0f1", height=2)
        sep.pack(fill="x", padx=24, pady=(0, 16))

        ctk.CTkLabel(info_panel, text="Key Points",
                     font=("Inter", 18, "bold"), text_color=cat_color,
                     anchor="w").pack(anchor="w", padx=24)

        for point in technique["key_points"]:
            row_f = ctk.CTkFrame(info_panel, fg_color="transparent")
            row_f.pack(anchor="w", padx=24, pady=3, fill="x")
            ctk.CTkLabel(row_f, text="▸", font=("Inter", 18, "bold"),
                         text_color=cat_color, width=20).pack(side="left")
            ctk.CTkLabel(row_f, text=point, font=("Inter", 18),
                         text_color=COLOR_TEXT, anchor="w",
                         wraplength=440, justify="left").pack(side="left", padx=6)

        # Right panel — image placeholder + actions
        right_panel = ctk.CTkFrame(self.video_canvas, fg_color="white",
                                   corner_radius=24, width=460, height=620)
        right_panel.pack_propagate(False)
        self.add_widget(cx + 255, cy, right_panel)

        # Image panel with Front / Left / Right viewpoint tabs
        media_box = ctk.CTkFrame(right_panel, fg_color="#ecf0f1",
                                 corner_radius=16, width=400, height=280,
                                 cursor="hand2")
        media_box.pack(padx=30, pady=(16, 0))
        media_box.pack_propagate(False)

        self._lesson_gif_frames = {}   # viewpoint → list of (CTkImage, delay_ms)
        self._lesson_thumb_refs = {}   # keep refs to prevent GC
        self._lesson_active_vp = technique["viewpoint"].lower()
        self._lesson_gif_anim_id = None  # after() id for cancellation
        self._lesson_gif_frame_idx = 0
        self._lesson_technique_key = technique["key"]
        self._lesson_loading_vp = None   # track which vp is being loaded

        # Image display label inside media_box
        media_img_label = ctk.CTkLabel(media_box, text="Loading...",
                                       font=("Inter", 18), image=None)
        media_img_label.pack(expand=True)

        # Tab strip (must be defined before _switch_vp so closure can reference tab_labels)
        tab_strip = ctk.CTkFrame(right_panel, fg_color="transparent")
        tab_strip.pack(pady=(6, 0))
        tab_labels = []

        def _animate_gif():
            """Cycle through GIF frames for the active viewpoint."""
            frames = self._lesson_gif_frames.get(self._lesson_active_vp, [])
            if not frames:
                return
            self._lesson_gif_frame_idx = self._lesson_gif_frame_idx % len(frames)
            ctk_img, delay = frames[self._lesson_gif_frame_idx]
            try:
                media_img_label.configure(image=ctk_img, text="")
            except Exception:
                return  # widget destroyed
            self._lesson_gif_frame_idx += 1
            self._lesson_gif_anim_id = self.after(delay, _animate_gif)

        def _on_frames_loaded(vp, frames):
            """Callback on main thread once background loading finishes."""
            if not frames:
                return
            self._lesson_gif_frames[vp] = frames
            self._lesson_thumb_refs[vp] = frames[0][0]
            # Only start animation if this viewpoint is still the active one
            if self._lesson_active_vp == vp:
                self._lesson_gif_frame_idx = 0
                media_img_label.configure(image=frames[0][0], text="")
                _animate_gif()

        def _load_vp_async(vp):
            """Load GIF frames for a viewpoint in a background thread."""
            if vp in self._lesson_gif_frames:
                # Already loaded — just start animating
                _on_frames_loaded(vp, self._lesson_gif_frames[vp])
                return
            self._lesson_loading_vp = vp
            media_img_label.configure(image=None, text="Loading...", font=("Inter", 18))
            gif_name = _lesson_gif_filename(self._lesson_technique_key, vp)
            if not gif_name:
                media_img_label.configure(text="\U0001f5bc\ufe0f", font=("Inter", 60))
                return
            gif_path = get_resource_path(f"lesson/{vp}_gif/{gif_name}")

            def _bg_load():
                frames = _load_gif_frames(gif_path, (380, 240))
                # Schedule callback on main thread
                try:
                    self.after(0, lambda: _on_frames_loaded(vp, frames))
                except Exception:
                    pass  # app closed during loading

            threading.Thread(target=_bg_load, daemon=True).start()

        def _switch_vp(vp: str):
            self._lesson_active_vp = vp
            self._lesson_gif_frame_idx = 0
            # Cancel existing animation
            if self._lesson_gif_anim_id is not None:
                self.after_cancel(self._lesson_gif_anim_id)
                self._lesson_gif_anim_id = None
            # Lazy-load this viewpoint
            _load_vp_async(vp)
            for lbl, bvp in tab_labels:
                lbl.configure(
                    fg_color=cat_color if bvp == vp else "#dfe6e9",
                    text_color="white" if bvp == vp else COLOR_TEXT
                )

        for vp_label, vp_key in [("Front", "front"), ("Left", "left"), ("Right", "right")]:
            lbl = ctk.CTkLabel(tab_strip, text=vp_label, font=("Inter", 15, "bold"),
                               width=90, height=32, corner_radius=8, cursor="hand2",
                               fg_color="#dfe6e9", text_color=COLOR_TEXT)
            lbl.pack(side="left", padx=4)
            lbl.bind("<Button-1>", lambda e, v=vp_key: _switch_vp(v))
            tab_labels.append((lbl, vp_key))

        # Seed initial viewpoint (lazy-load)
        _switch_vp(self._lesson_active_vp)

        # Bind image box click to zoom
        for w in (media_box, media_img_label):
            w.bind("<Button-1>", lambda e: self._show_image_zoom())

        ctk.CTkLabel(right_panel, text="Click image to enlarge",
                     font=("Inter", 13), text_color="gray").pack(pady=(4, 0))

        ctk.CTkButton(right_panel, text="Let's Practise! →",
                      font=("Inter", 22, "bold"),
                      fg_color=COLOR_SUCCESS, hover_color="#2ecc71",
                      height=60, corner_radius=16,
                      command=self.start_lesson_practice).pack(padx=30, pady=(16, 0), fill="x")

        ctk.CTkButton(right_panel, text="← Back",
                      font=("Inter", 18),
                      fg_color="transparent", border_width=2,
                      border_color=COLOR_TEXT, text_color=COLOR_TEXT,
                      hover_color="#ecf0f1", height=44, corner_radius=22,
                      command=self.show_lesson_select).pack(padx=30, pady=(10, 24), fill="x")

    def _show_image_zoom(self):
        """Overlay a full-screen-height animated GIF panel on top of the instruction screen."""
        cx = self.screen_width // 2
        sh = self.screen_height

        # Semi-transparent backdrop
        overlay_bg = self.video_canvas.create_rectangle(
            0, 0, self.screen_width, sh,
            fill="#000000", stipple="gray50", outline=""
        )
        self.canvas_items.append(overlay_bg)

        # Zoom panel
        zoom_panel = ctk.CTkFrame(self.video_canvas, fg_color="#ecf0f1",
                                  corner_radius=20, width=480, height=sh - 40)
        zoom_panel.pack_propagate(False)

        # Load animated GIF for zoom view — use whichever tab is active
        self._lesson_zoom_refs = []  # keep frame refs alive
        self._lesson_zoom_anim_id = None
        technique_key = self.current_lesson.get("key") if self.current_lesson else None
        active_vp = getattr(self, "_lesson_active_vp", "front")
        gif_name = _lesson_gif_filename(technique_key, active_vp) if technique_key else None

        zoom_img_label = ctk.CTkLabel(zoom_panel, text="Loading...",
                                       font=("Inter", 24), image=None)
        zoom_img_label.pack(expand=True, pady=(16, 0))

        if gif_name:
            gif_path = get_resource_path(f"lesson/{active_vp}_gif/{gif_name}")
            max_w, max_h = 440, sh - 140

            def _bg_load_zoom():
                zoom_frames = _load_gif_frames(gif_path, (max_w, max_h))
                try:
                    self.after(0, lambda: _start_zoom_anim(zoom_frames))
                except Exception:
                    pass

            def _start_zoom_anim(zoom_frames):
                if zoom_frames:
                    self._lesson_zoom_refs = [f[0] for f in zoom_frames]
                    zoom_frame_idx = [0]

                    def _animate_zoom():
                        idx = zoom_frame_idx[0] % len(zoom_frames)
                        ctk_img, delay = zoom_frames[idx]
                        try:
                            zoom_img_label.configure(image=ctk_img, text="")
                        except Exception:
                            return  # widget destroyed
                        zoom_frame_idx[0] = idx + 1
                        self._lesson_zoom_anim_id = self.after(delay, _animate_zoom)

                    _animate_zoom()
                else:
                    zoom_img_label.configure(text="\U0001f5bc\ufe0f", font=("Inter", 160))

            threading.Thread(target=_bg_load_zoom, daemon=True).start()
        else:
            zoom_img_label.configure(text="\U0001f5bc\ufe0f", font=("Inter", 160))

        technique_name = self.current_lesson["name"] if self.current_lesson else ""
        vp_display = active_vp.title()
        ctk.CTkLabel(zoom_panel, text=f"{technique_name}  ({vp_display} view)",
                     font=("Inter", 20, "bold"), text_color="#2c3e50").pack(pady=(8, 0))

        ctk.CTkButton(zoom_panel, text="✕  Close",
                      font=("Inter", 18, "bold"),
                      fg_color=COLOR_TEXT, hover_color="#555",
                      text_color="white", height=48, corner_radius=12,
                      command=lambda: self._close_image_zoom(overlay_bg, zoom_win)
                      ).pack(padx=40, pady=(0, 24), fill="x")

        zoom_win = self.video_canvas.create_window(cx, sh // 2, window=zoom_panel, anchor="center")
        self.canvas_items.append(zoom_win)

        # Also close when clicking the backdrop
        self.video_canvas.tag_bind(overlay_bg, "<Button-1>",
                                   lambda e: self._close_image_zoom(overlay_bg, zoom_win))

    def _close_image_zoom(self, overlay_bg, zoom_win):
        """Remove the zoom overlay items from the canvas and stop zoom animation."""
        if getattr(self, '_lesson_zoom_anim_id', None) is not None:
            self.after_cancel(self._lesson_zoom_anim_id)
            self._lesson_zoom_anim_id = None
        try:
            self.video_canvas.delete(overlay_bg)
            self.video_canvas.delete(zoom_win)
        except Exception:
            pass



    def start_lesson_practice(self):
        """Begin the lesson practice loop using the existing zoning → countdown → feedback flow."""
        tech = self.current_lesson
        # Set up a single-user guest session with the full structure start_zoning_check expects
        self.num_users = 1
        # Keep the UI-cased string ("Front", "Right Side", "Left Side") so that
        # analyze_zones and show_feedback can map it correctly via viewpoint_mapping.
        viewpoint = tech.get("viewpoint", "Front")  # e.g. "Front", NOT "front"
        self.user_configs = [{
            "name": "Practitioner",
            "viewpoint": viewpoint,
            "user": None,       # start_zoning_check creates a guest if None
            "session_id": None, # filled in by start_zoning_check
            "zone": None,
        }]
        self.start_zoning_check()


    def show_user_count(self):
        self.app_state = AppState.USER_COUNT
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)
        
        cx, cy = self.screen_width // 2, self.screen_height // 2
        self.add_text(cx, cy - 120, "How many masters today?", font=FONT_HEADER, fill=COLOR_TEXT)
        
        bg_frame = ctk.CTkFrame(self.video_canvas, fg_color="transparent")
        for i in range(1, 4):
            btn = ctk.CTkButton(bg_frame, text=str(i), font=("Inter", 64, "bold"), width=150, height=150,
                              corner_radius=25, fg_color="#fff", text_color=COLOR_ACCENT,
                              hover_color="#ecf0f1",
                              command=lambda n=i: self.start_config(n))
            btn.pack(side="left", padx=30)
        self.add_widget(cx, cy + 50, bg_frame)
        
        # Back Button — return to Mode Select
        btn_back = ctk.CTkButton(self.video_canvas, text="← Back", font=("Inter", 18),
                                fg_color="transparent", border_width=2, border_color="white", text_color="white",
                                hover_color="#aecef7", height=44, width=160, corner_radius=22,
                                command=self.show_mode_select)
        self.add_widget(cx, cy + 250, btn_back)

    def start_config(self, n):
        self.num_users = n
        self.user_configs = []
        for i in range(n):
            self.user_configs.append({
                'user': None,
                'viewpoint': ctk.StringVar(value="Front"),
                'session_id': None
            })
        self.show_config_screen()

    def show_config_screen(self):
        self.app_state = AppState.CONFIG
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)
        
        col_w = self.screen_width // self.num_users
        
        for i in range(1, self.num_users):
            self.canvas_items.append(
                self.video_canvas.create_line(i*col_w, 100, i*col_w, self.screen_height-150, fill="white", width=2, dash=(10,10))
            )

        for i in range(self.num_users):
            cx = (i * col_w) + (col_w // 2)
            cy = self.screen_height // 2
            
            # Card (White on light blue) - Simplified without target pose selection
            card = ctk.CTkFrame(self.video_canvas, fg_color="white", corner_radius=20, width=350, height=320)
            card.pack_propagate(False)
            
            current_name = self.user_configs[i]['user']['name'] if self.user_configs[i]['user'] else f"Guest {i+1}"
            btn_user = ctk.CTkButton(card, text=current_name, font=("Inter", 28, "bold"),
                                    fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, height=60, corner_radius=10,
                                    text_color="white",
                                    command=lambda idx=i: self.open_user_select(idx))
            btn_user.pack(pady=(30, 20), padx=20, fill="x")
            
            ctk.CTkLabel(card, text="Click to change user", font=("Inter", 14), text_color="gray").pack(pady=(0, 20))
            
            ctk.CTkLabel(card, text="Viewpoint", font=("Inter", 18, "bold"), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
            ctk.CTkSegmentedButton(card, values=["Front", "Right Side", "Left Side"], variable=self.user_configs[i]['viewpoint'], 
                                  font=("Inter", 16), fg_color="#ecf0f1", selected_color=COLOR_ACCENT, selected_hover_color=COLOR_ACCENT_HOVER, text_color="black").pack(pady=(5, 20), padx=20, fill="x")
            
            # Ready indicator - No target pose selection needed
            ctk.CTkLabel(card, text="✓ Ready for Recognition", font=("Inter", 18, "bold"), text_color=COLOR_SUCCESS).pack(pady=(20, 0))
            
            self.add_widget(cx, cy, card)

        btn = ctk.CTkButton(self.video_canvas, text="LOCK IN [ENTER]", font=("Inter", 32, "bold"),
                           fg_color=COLOR_SUCCESS, hover_color="#27ae60", height=90, width=400, corner_radius=45,
                           command=self.start_zoning_check)
        self.add_widget(self.screen_width//2, self.screen_height - 100, btn)
        
        # Back Button
        btn_back = ctk.CTkButton(self.video_canvas, text="← Back", font=("Inter", 18),
                                fg_color="transparent", border_width=2, border_color="white", text_color="white",
                                hover_color="#aecef7", height=44, width=160, corner_radius=22,
                                command=self.show_user_count)
        self.add_widget(120, self.screen_height - 60, btn_back)

    def open_user_select(self, slot_index):
        dialog = UserManagementDialog(self, self.db)
        self.wait_window(dialog.dialog)
        
        selected_user = dialog.selected_user
        if selected_user:
            self.user_configs[slot_index]['user'] = selected_user
            self.show_config_screen()

    def restart_zoning(self):
        """Restart the zoning phase for a new repetition"""
        self.app_state = AppState.ZONING
        self.clear_ui()
        self.frozen_frame = None
        self.analysis_results = {}
        self.realtime_pose_cache.clear()  # drop stale per-zone pose data
        self.show_user_names = True
        self.names_shown_time = time.time()
        self.zoning_start_time = time.time()
        # Clear all GCN / stick / smoothing caches so the warm-up check
        # starts fresh and deferral doesn't hang on the next rep.
        if self.pose_analyzer:
            self.pose_analyzer.clear_session_cache()
        self.after(200, self.check_zones_and_countdown)

    def start_zoning_check(self):
        for config in self.user_configs:
            user_id = config['user']['id'] if config['user'] else None
            if not user_id:
                guest_name = f"Guest_{int(time.time())}_{np.random.randint(100)}"
                user_id = self.db.create_user(guest_name)
                config['user'] = self.db.get_user_by_id(user_id)
            
            # In guided lesson mode, record the target technique for analytics
            target_pose = self.current_lesson["key"] if self.current_lesson else None
            sid = self.db.start_session(user_id, target_pose=target_pose)
            config['session_id'] = sid

        # Clear stale caches from any previous session so the GCN warm-up
        # check doesn't see a leftover _cached_g_feat and so inference
        # starts fresh without leaked predictions or stick history.
        self.realtime_pose_cache.clear()
        if self.pose_analyzer:
            self.pose_analyzer.clear_session_cache()

        self.app_state = AppState.ZONING
        self.clear_ui()
        self.show_user_names = True
        self.names_shown_time = time.time()
        # Track zoning start time for 5-second timeout
        self.zoning_start_time = time.time()
        # Check if users are in zones before starting countdown
        self.check_zones_and_countdown()
    
    def check_zones_and_countdown(self):
        """Validate that users are properly positioned in zones before starting countdown"""
        if self.app_state != AppState.ZONING:
            return
        
        # Check for 5-second timeout (was 10s)
        elapsed_time = time.time() - self.zoning_start_time
        if elapsed_time > 5.0:
            # Timeout: start countdown anyway
            self.after(1000, self.start_countdown)
            return
        
        # Check person positioning in each zone
        if self.current_frame is not None and self.pose_analyzer:
            h, w, _ = self.current_frame.shape
            col_w = w // self.num_users
            properly_positioned = 0
            
            for i in range(self.num_users):
                x_start = i * col_w
                x_end = (i + 1) * col_w
                zone_frame = self.current_frame[:, x_start:x_end].copy()
                zone_h, zone_w = zone_frame.shape[:2]
                
                try:
                    # OPTIMIZATION: Use cached result from update_feed if available
                    person_data = self.realtime_pose_cache.get(i)
                    
                    if person_data is None:
                        # Fallback: Run detection if not in cache
                        # MODE: video - Live camera preview (temporal smoothing OK)
                        results = self.pose_analyzer.process_frame(zone_frame, skip_ml_inference=True, skip_stick_detection=False)
                        if results and len(results) > 0:
                            person_data = results[0]

                    if person_data:
                        landmarks = person_data.get('landmarks_absolute')
                        
                        if landmarks and len(landmarks) >= 33:
                            # Use the same full-body check as live feedback for consistency
                            if self.feedback_analyzer:
                                is_vis, _, _, _ = self.feedback_analyzer.is_full_body_visible(
                                    landmarks, zone_w, zone_h, min_visible=7
                                )
                                if is_vis:
                                    properly_positioned += 1
                            else:
                                # Fallback if feedback_analyzer not ready
                                visible_count = sum(
                                    1 for idx in [0, 11, 12, 23, 24, 27, 28]
                                    if idx < len(landmarks)
                                    and 0 < landmarks[idx][0] < zone_w
                                    and 0 < landmarks[idx][1] < zone_h
                                )
                                if visible_count >= 5:
                                    properly_positioned += 1
                except:
                    pass
            
            # If all users properly positioned, start countdown
            if properly_positioned >= self.num_users:
                self.after(1000, self.start_countdown)
            else:
                # Check again in 500ms
                self.after(500, self.check_zones_and_countdown)
        else:
            # Fallback: check again
            self.after(500, self.check_zones_and_countdown)

    def start_countdown(self):
        self.app_state = AppState.COUNTDOWN
        self.clear_ui()
        self.countdown_timer = 5
        
        cx, cy = self.screen_width//2, self.screen_height//2
        
        # Draw circle background
        radius = 150
        self.countdown_circle_id = self.video_canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=COLOR_WARNING, outline="", tags="countdown_circle"
        )
        self.canvas_items.append(self.countdown_circle_id)
        
        self.count_text_id = self.video_canvas.create_text(cx, cy, text="5", font=("Inter", 200, "bold"), fill="white")
        self.canvas_items.append(self.count_text_id)

        # Kick off a background GCN warm-up inference so _cached_g_feat is
        # set well before SNAP fires (5-second runway).
        self._snapshot_defer_count = 0
        threading.Thread(target=self._warmup_gcn, daemon=True).start()

        self.update_countdown()

    def _warmup_gcn(self):
        """Run one GCN inference in a background thread during countdown.

        This populates _cached_g_feat / _cached_prediction so that
        capture_snapshot() can proceed without deferring.
        """
        try:
            frame = self.current_frame
            if frame is None or self.pose_analyzer is None:
                return
            h, w = frame.shape[:2]
            col_w = w // self.num_users
            # Use zone 0; any zone is fine — we just need GCN to run once.
            zone_frame = frame[:, 0:col_w].copy()
            viewpoint = "front"  # default; exact viewpoint doesn't matter for warm-up
            if self.pose_analyzer.gcn_engine:
                self.pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            # MODE: snapshot - GCN warm-up inference (single image, no temporal smoothing needed)
            self.pose_analyzer.process_frame(
                zone_frame, skip_ml_inference=False, mode='snapshot'
            )
            print("[GCN-WARMUP] Warm-up inference complete — GCN ready.")
        except Exception as e:
            print(f"[GCN-WARMUP] Warm-up inference failed (non-fatal): {e}")


    def update_countdown(self):
        if self.app_state != AppState.COUNTDOWN: return

        if self.countdown_timer > 0:
            self.video_canvas.itemconfig(self.count_text_id, text=str(self.countdown_timer))
            self.countdown_timer -= 1
            self.after(1000, self.update_countdown)
        else:
            # Change circle to green for SNAP
            if self.countdown_circle_id:
                self.video_canvas.itemconfig(self.countdown_circle_id, fill=COLOR_SUCCESS)
            self.video_canvas.itemconfig(self.count_text_id, text="SNAP!")
            self.capture_snapshot()

    def capture_snapshot(self):
        # Guard: ensure GCN has completed at least one inference so global_features
        # will be available for hybrid feedback corrections.
        gcn_ready = (
            self.pose_analyzer is not None
            and hasattr(self.pose_analyzer, '_cached_g_feat')
        )
        if not gcn_ready:
            # Cap deferrals at 2 (max 2 extra seconds) so we never hang forever.
            defer_count = getattr(self, '_snapshot_defer_count', 0)
            if defer_count < 2:
                self._snapshot_defer_count = defer_count + 1
                if self.count_text_id:
                    try:
                        self.video_canvas.itemconfig(self.count_text_id, text="...", font=("Inter", 80, "bold"))
                        if self.countdown_circle_id:
                            self.video_canvas.itemconfig(self.countdown_circle_id, fill=COLOR_WARNING)
                    except Exception:
                        pass
                print(f"[SNAPSHOT] GCN not warmed up yet — deferring 1 s ({self._snapshot_defer_count}/2)")
                self.after(1000, self.capture_snapshot)
                return
            else:
                # Proceed anyway — don't let a slow warm-up block the user indefinitely.
                print("[SNAPSHOT] Defer limit reached — proceeding without warm-up cache.")

        self.app_state = AppState.SNAPSHOT
        if self.current_frame is not None:
            self.frozen_frame = self.current_frame.copy()
            # Analyze poses in the snapshot
            if self.pose_analyzer:
                self.analysis_results = self.analyze_zones(self.frozen_frame)
        self.after(800, self.show_feedback)


    def show_feedback(self):
        self.app_state = AppState.FEEDBACK
        self.clear_ui()
        self.feedback_timer = 10  # 10 s so users have time to read all corrections
        
        # Use letterbox-aware coordinates so text aligns with the actual video zones
        vx = self.video_x_offset
        vy = self.video_y_offset
        vw = self.video_display_width
        vh = self.video_display_height
        
        col_w = vw // self.num_users
        video_bottom = vy + vh  # Bottom edge of the actual video area
        
        for i in range(self.num_users):
            cx = vx + (i * col_w) + (col_w // 2)
            config = self.user_configs[i]
            
            # Get GCN analysis results for this zone
            zone_result = self.analysis_results.get(i, {})
            predicted_class = zone_result.get('predicted_class', 'N/A')
            confidence = zone_result.get('confidence', 0.0)
            stick_detected = zone_result.get('stick_detected', False)
            
            # Pure recognition mode - confidence-based scoring only
            # Get viewpoint-specific confidence threshold from gcn_model_config.json (single source of truth)
            vp_raw = config['viewpoint']
            viewpoint_ui = vp_raw.get() if hasattr(vp_raw, 'get') else vp_raw
            viewpoint_mapping = {
                "Front": "front",
                "Right Side": "right",
                "Left Side": "left"
            }
            viewpoint = viewpoint_mapping.get(viewpoint_ui, viewpoint_ui).lower()
            gcn_config = self.pose_analyzer.gcn_engine.config if (self.pose_analyzer and self.pose_analyzer.gcn_engine) else {}
            threshold = gcn_config.get('models', {}).get(viewpoint, {}).get('confidence_threshold', 0.55)

            # Scoring bands relative to per-viewpoint threshold
            # gcn_inference.py already filters below threshold, so confidence > 0 means >= threshold
            high_confidence = (confidence >= threshold + 0.15)  # Excellent band
            good_confidence  = (confidence >= threshold)         # Good band (minimum)
            pose_detected = (predicted_class != 'N/A' and predicted_class.lower() != 'no technique detected' and confidence > 0)
            
            # ─────────────────────────────────────────────────────────────
            # SCORE + FEEDBACK — two completely different paths
            # ─────────────────────────────────────────────────────────────
            feedback_messages = []

            if self.current_lesson:
                # ── GUIDED MODE ──────────────────────────────────────────
                target_key  = self.current_lesson["key"]
                correct_hit = pose_detected and (predicted_class == target_key)
                wrong_hit   = pose_detected and (predicted_class != target_key)

                # Always compare against target form (even if nothing detected)
                if self.feedback_analyzer:
                    analysis = self.feedback_analyzer.analyze(
                        result=zone_result,
                        target_form=target_key,
                        confidence_threshold=threshold,
                        viewpoint=viewpoint,
                        gcn_engine=self.pose_analyzer.gcn_engine if self.pose_analyzer else None
                    )
                    # No cap — show every correction the system found
                    prioritized = self.feedback_analyzer.get_prioritized_messages(analysis, max_messages=10)
                    feedback_messages = [msg for msg, t in prioritized if t in ('error', 'warning')]
                    if not feedback_messages and correct_hit and high_confidence:
                        suggestions = [msg for msg, t in prioritized if t == 'suggestion']
                        if suggestions:
                            feedback_messages = [suggestions[0]]
                    print(f"[FEEDBACK][GUIDED] target={target_key} | is_correct={analysis.get('is_correct')} | conf={confidence:.2f} | errors={analysis.get('errors',[])} | warnings={analysis.get('warnings',[])} | msgs={feedback_messages}")

                target_display = target_key.replace('_correct', '').replace('_', ' ').title()

                if wrong_hit:
                    color = "#e67e22"
                    detected_display = predicted_class.replace('_correct', '').replace('_', ' ').title()
                    score_text = "WRONG TECHNIQUE"
                    # Show ALL real corrections toward the target form — no cap
                    if not feedback_messages:
                        feedback_messages = [f"Adjust to {target_display} position"]
                elif correct_hit and high_confidence:
                    color = "#2ecc71"
                    score_text = "EXCELLENT!"
                elif correct_hit:
                    color = "#bfff00"
                    score_text = "GOOD"
                    # Show corrections even on a good hit — they still have room to improve
                    if not feedback_messages:
                        feedback_messages = ["Almost there — refine your form"]
                else:
                    color = "#e74c3c"
                    score_text = "NOT DETECTED"
                    # Show corrections if available, else a useful directional hint
                    if not feedback_messages:
                        feedback_messages = [f"Get into {target_display} position", "Face the camera fully"]

                # is_correct flag for DB: only true when the right technique was hit well
                is_correct_db = correct_hit and high_confidence
                # Track rep success so _show_lesson_feedback_end knows whether to celebrate
                # or silently restart. Use a per-user flag keyed by zone index.
                if i == 0:  # Only the first zone drives the lesson gate
                    self.last_attempt_success = is_correct_db

            else:
                # ── FREE PRACTICE MODE ───────────────────────────────────
                if not pose_detected:
                    color = "#e74c3c"
                    score_text = "NOT DETECTED"
                elif high_confidence:
                    color = "#2ecc71"
                    score_text = "EXCELLENT!"
                elif good_confidence:
                    color = "#ffff00"
                    score_text = "GOOD"
                else:
                    color = "#f39c12"
                    score_text = "FAIR"

                if self.feedback_analyzer:
                    # In free practice, compare against the predicted class.
                    # Use a raised threshold so we still surface corrections
                    # even when confidence is high — corrections are the main goal.
                    fp_threshold = min(threshold + 0.15, 0.98)
                    analysis = self.feedback_analyzer.analyze(
                        result=zone_result,
                        target_form=predicted_class,
                        confidence_threshold=fp_threshold,
                        viewpoint=viewpoint,
                        gcn_engine=self.pose_analyzer.gcn_engine if self.pose_analyzer else None
                    )
                    prioritized = self.feedback_analyzer.get_prioritized_messages(analysis, max_messages=10)
                    # Always show errors/warnings first; fall back to suggestions only if nothing else
                    feedback_messages = [msg for msg, t in prioritized if t in ('error', 'warning')]
                    if not feedback_messages:
                        feedback_messages = [msg for msg, t in prioritized if t == 'suggestion']
                    print(f"[FEEDBACK][FREE]   predicted={predicted_class} | is_correct={analysis.get('is_correct')} | conf={confidence:.2f} | errors={analysis.get('errors',[])} | warnings={analysis.get('warnings',[])} | msgs={feedback_messages}")

                # DEMO OVERRIDE: Force specific feedback messages
                if DEMO_MODE and getattr(self, '_demo_image_name', None):
                    forced = getattr(self, '_demo_forced_feedback', None)
                    if forced is not None:
                        feedback_messages = forced
                        print(f"[DEMO] Forced feedback for {self._demo_image_name}: {feedback_messages}")

                is_correct_db = high_confidence
            
            # Save performance
            if config['session_id']:
                self.db.save_performance(
                    session_id=config['session_id'],
                    user_id=config['user']['id'],
                    pose_detected=predicted_class,
                    confidence=confidence,
                    is_correct=is_correct_db,
                    stick_detected=stick_detected
                )
            
            # Display user name (positioned relative to video bottom edge)
            self.add_text(cx, video_bottom - 280, config['user']['name'], font=("Inter", 24, "bold"), fill="white")
            
            # Display detected pose name (if recognized)
            if pose_detected:
                # Convert technical name to display name
                display_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
                self.add_text(cx, video_bottom - 230, display_name, font=("Inter", 20), fill="white")
            
            # Display confidence/status score
            # When hints exist and it's not excellent, shrink the score and let hints dominate
            is_excellent = (color == "#2ecc71")
            score_font = ("Inter", 48, "bold") if (is_excellent or not feedback_messages) else ("Inter", 28, "bold")
            self.add_text(cx, video_bottom - 185, score_text, font=score_font, fill=color)

            # --- DISPLAY CORRECTIVE FEEDBACK ---
            # Stick indicator is fixed at the very bottom; corrections grow UPWARD from above it.
            stick_text = "✓ Stick" if stick_detected else "✗ No stick"
            stick_color = "#27ae60" if stick_detected else "#95a5a6"
            self.add_text(cx, video_bottom - 30, stick_text, font=("Inter", 16), fill=stick_color)

            if feedback_messages:
                hint_font    = ("Inter", 20, "bold") if is_excellent else ("Inter", 22, "bold")
                hint_color   = "white"
                hint_spacing = 30  # px between lines
                # Start just above the stick indicator and grow upward
                y_base = video_bottom - 60
                for msg in reversed(feedback_messages):
                    self.add_text(cx, y_base, msg, font=hint_font, fill=hint_color)
                    y_base -= hint_spacing
            else:
                # No hints: show confidence % as secondary info
                if pose_detected:
                    percentage = f"{int(confidence * 100)}%"
                    self.add_text(cx, video_bottom - 70, percentage, font=("Inter", 24), fill="white")
        
        # Timer label (must exist before update_feedback_timer tries to itemconfig it)
        self.timer_text_id = self.video_canvas.create_text(
            self.screen_width // 2, 50,
            text=f"Next in {self.feedback_timer}...",
            font=("Inter", 28), fill="white"
        )
        self.canvas_items.append(self.timer_text_id)

        # FINISH button (top-right corner)
        btn_finish = ctk.CTkButton(
            self.video_canvas, text="FINISH", width=150, height=50,
            fg_color=COLOR_WARNING, font=("Inter", 18, "bold"),
            command=self.end_session_and_show_results
        )
        self.add_widget(self.screen_width - 100, 50, btn_finish)

        self.update_feedback_timer()

    def generate_form_feedback(self, target_class, viewpoint, live_angles, landmarks=None):
        """Generate feedback based on angle deviations from template"""
        # Template keys are like "front_crown_thrust_correct"
        template_key = f"{viewpoint}_{target_class}"
        
        deviations = []
        
        # 1. Stance & Posture Heuristics
        if landmarks:
             # Handle MediaPipe NormalizedLandmarkList or list of objs
             lms = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
             if len(lms) > 28:
                 # Stance Width (X-distance between ankles)
                 left_ankle = lms[27]
                 right_ankle = lms[28]
                 width = abs(left_ankle.x - right_ankle.x)
                 if width < 0.05: # Heuristic for very narrow stance
                     deviations.append("Widen stance")
        
        # 2. Angle Deviations
        # Config: name, min_msg (if user < range), max_msg (if user > range)
        angle_map = {
            'right_elbow_angle': {'name': 'Right Arm', 'min': 'Extend', 'max': 'Bend'}, # <180 is bent
            'left_elbow_angle': {'name': 'Left Arm', 'min': 'Extend', 'max': 'Bend'},
            'right_shoulder_angle': {'name': 'Right Arm', 'min': 'Raise', 'max': 'Lower'}, # 0=Down, 180=Up
            'left_shoulder_angle': {'name': 'Left Arm', 'min': 'Raise', 'max': 'Lower'},
            'right_knee_angle': {'name': 'Right Knee', 'min': 'Straighten', 'max': 'Bend'}, # <180 is bent
            'left_knee_angle': {'name': 'Left Knee', 'min': 'Straighten', 'max': 'Bend'}
        }
        
        if template_key in self.feature_templates:
            template = self.feature_templates[template_key]
            for angle_key, config in angle_map.items():
                # Fix: live_angles uses keys like 'right_elbow', template uses 'right_elbow_angle'
                live_key = angle_key.replace('_angle', '')
                
                if angle_key in template and live_key in live_angles:
                    user_val = live_angles[live_key]
                    t_stats = template[angle_key]
                    t_mean = t_stats['mean']
                    t_std = t_stats['std']
                    
                    # 1.2 std dev tolerance
                    min_val = t_mean - (1.2 * t_std)
                    max_val = t_mean + (1.2 * t_std)
                    
                    if user_val < min_val:
                        deviations.append(f"{config['min']} {config['name']}")
                    elif user_val > max_val:
                        deviations.append(f"{config['max']} {config['name']}")
        
        if not deviations:
            if template_key not in self.feature_templates:
                 return f"Goal: {target_class.replace('_', ' ').title()}"
            return "Good form!"
            
        # Return unique top tips
        unique_tips = sorted(list(set(deviations)))
        return "Tip: " + ", ".join(unique_tips[:2])

    def show_results_screen(self):
        self.app_state = AppState.RESULTS
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)
        
        cx = self.screen_width // 2
        self.add_text(cx, 80, "Session Complete!", font=("Inter", 64, "bold"), fill="white")
        
        container = ctk.CTkFrame(self.video_canvas, fg_color="transparent")
        self.add_widget(cx, 350, container)
        
        for i in range(self.num_users):
            user = self.user_configs[i]['user']
            zone_result = self.analysis_results.get(i, {})
            predicted_class = zone_result.get('predicted_class', 'N/A')
            confidence = zone_result.get('confidence', 0.0)
            
            # Filter out neutral_stance (training buffer)
            if predicted_class.lower() == 'neutral_stance':
                predicted_class = 'No Technique Detected'
                confidence = 0.0
            
            # Create gradient-style card
            card = ctk.CTkFrame(container, fg_color=("#1a1f36", "#1a1f36"), width=320, height=450, corner_radius=25)
            card.pack(side="left", padx=25)
            card.pack_propagate(False)
            
            # Header Section
            header_frame = ctk.CTkFrame(card, fg_color=COLOR_ACCENT, height=100, corner_radius=20)
            header_frame.pack(fill="x", padx=15, pady=15)
            header_frame.pack_propagate(False)
            
            ctk.CTkLabel(header_frame, text=user['name'], font=("Inter", 32, "bold"), text_color="white").pack(pady=25)
            
            # Result Section
            result_frame = ctk.CTkFrame(card, fg_color="transparent")
            result_frame.pack(fill="x", padx=20, pady=20)
            
            ctk.CTkLabel(result_frame, text="Detected Technique:", font=("Inter", 16), text_color="#9ba3c0").pack()
            ctk.CTkLabel(result_frame, text=predicted_class.replace('_', ' ').title(), 
                        font=("Inter", 22, "bold"), text_color="white", wraplength=250).pack(pady=(5, 15))
            
            ctk.CTkLabel(result_frame, text=f"Confidence: {confidence*100:.1f}%", 
                        font=("Inter", 18), text_color="#67e0a3").pack(pady=5)
            
            # Status Badge
            status_frame = ctk.CTkFrame(card, fg_color="#27ae60", corner_radius=10)
            status_frame.pack(pady=10)
            ctk.CTkLabel(status_frame, text="✓ Data Saved", font=("Inter", 16, "bold"), text_color="white").pack(padx=20, pady=8)
            
            # View History Button
            btn_details = ctk.CTkButton(card, text="VIEW HISTORY", font=("Inter", 16, "bold"),
                                       fg_color="#2c3e50", hover_color="#34495e", height=50, width=240, corner_radius=15,
                                       command=lambda u=user: self.open_full_results(u))
            btn_details.pack(pady=15)
        
        btn_restart = ctk.CTkButton(self.video_canvas, text="NEW SESSION", font=("Inter", 28, "bold"),
                                   fg_color=COLOR_SUCCESS, hover_color="#27ae60", height=80, width=350, corner_radius=40,
                                   command=self.show_splash)
        self.add_widget(cx, self.screen_height - 120, btn_restart)

    def open_full_results(self, user):
        rw = ResultsWindow(self, self.db, user)
    
    def update_feedback_timer(self):
        """Update the feedback timer countdown"""
        if self.app_state != AppState.FEEDBACK:
            return
        
        if self.feedback_timer > 0:
            self.video_canvas.itemconfig(self.timer_text_id, text=f"Next in {self.feedback_timer}...")
            self.feedback_timer -= 1
            self.after(1000, self.update_feedback_timer)
        else:
            if self.current_lesson:
                # Lesson mode: let user choose to try again or pick another technique
                self._show_lesson_feedback_end()
            else:
                # Free practice: auto-restart zoning for the next rep
                self.restart_zoning()

    def _show_lesson_feedback_end(self):
        """Post-feedback screen shown in lesson mode.

        Only shows the 'Rep Complete!' celebration screen if the user
        actually hit the target pose at an acceptable confidence.
        If the attempt failed (wrong technique or not detected),
        it silently restarts the zoning phase so the user tries again.
        """
        succeeded = getattr(self, 'last_attempt_success', False)

        if not succeeded:
            # Attempt failed — restart immediately without any celebration screen.
            print("[LESSON] Attempt did not meet target — restarting zoning.")
            self.restart_zoning()
            return

        # Attempt succeeded — show the celebration / next-step screen.
        self.clear_ui()
        self.video_canvas.configure(bg=COLOR_BG)
        cx, cy = self.screen_width // 2, self.screen_height // 2

        tech_name = self.current_lesson.get("name", "Technique")
        self.add_text(cx, cy - 160, "Rep Complete!", font=("Inter", 48, "bold"), fill="white")
        self.add_text(cx, cy - 100, tech_name, font=("Inter", 28), fill="white")

        # Try Again
        btn_retry = ctk.CTkButton(
            self.video_canvas, text="🔄  Try Again",
            font=("Inter", 24, "bold"),
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            height=70, width=320, corner_radius=18,
            command=self.start_lesson_practice
        )
        self.add_widget(cx, cy, btn_retry)

        # Choose Another
        btn_other = ctk.CTkButton(
            self.video_canvas, text="← Choose Another Technique",
            font=("Inter", 20),
            fg_color="transparent", border_width=2,
            border_color="white", text_color="white",
            hover_color="#aecef7", height=56, width=320, corner_radius=18,
            command=self.show_lesson_select
        )
        self.add_widget(cx, cy + 100, btn_other)

        # Exit to splash
        btn_exit = ctk.CTkButton(
            self.video_canvas, text="Exit to Menu",
            font=("Inter", 18),
            fg_color="transparent", border_width=2,
            border_color="white", text_color="white",
            hover_color="#aecef7", height=48, width=220, corner_radius=14,
            command=self.show_splash
        )
        self.add_widget(cx, cy + 180, btn_exit)

    # --- VIDEO ENGINE ---

    def draw_vertical_separator(self, frame, x, h):
        cv2.line(frame, (x, 50), (x, h-50), (255, 255, 255), 2)

    def draw_pose_keypoints(self, frame, zone_results, col_w, cols, prediction_ready=False, success=False, use_individual_colors=False):
        """Draw pose keypoints and skeleton on the frame
        
        Args:
            prediction_ready: If True, analysis is complete.
            success: If True (and prediction_ready), draw green. If False, draw orange/red.
            use_individual_colors: If True, use per-zone 'is_correct' flag for colors (overrides success param)
        """
        if not self.pose_analyzer:
            return
        
        for i, zone_data in zone_results.items():
            # Use landmarks_absolute instead of landmarks (MediaPipe object)
            if 'landmarks_absolute' not in zone_data or not zone_data['landmarks_absolute']:
                continue
            
            landmarks_abs = zone_data['landmarks_absolute']
            
            # Determine color for this specific zone
            if use_individual_colors and 'is_correct' in zone_data:
                # Use per-zone status for granular color (Perfect=Green, Good=Yellow, Bad=Red)
                status = zone_data.get('status', 'bad')
                is_correct = zone_data.get('is_correct', False)
                
                if status == 'perfect':
                     skeleton_color = (0, 255, 0)    # Green - Perfect
                     keypoint_fill = (0, 255, 0)
                     keypoint_border = (0, 200, 0)
                elif status == 'good':
                     skeleton_color = (0, 255, 255)  # Yellow - Good
                     keypoint_fill = (0, 255, 255)
                     keypoint_border = (0, 200, 200)
                elif status == 'wrong':
                     skeleton_color = (0, 140, 255)  # Orange (BGR) - Wrong technique
                     keypoint_fill = (0, 140, 255)
                     keypoint_border = (0, 100, 200)
                elif is_correct:  # Fallback for boolean True without status
                     skeleton_color = (0, 255, 0)
                     keypoint_fill = (0, 255, 0)
                     keypoint_border = (0, 200, 0)
                else:
                     skeleton_color = (0, 0, 255)    # Red - Bad/Not detected
                     keypoint_fill = (0, 0, 255)
                     keypoint_border = (0, 0, 180)
            elif prediction_ready:
                # Use global success flag (existing behavior for backward compatibility)
                if success:
                    skeleton_color = (0, 255, 0)  # Green - Success/Correct
                    keypoint_fill = (0, 255, 0)
                    keypoint_border = (0, 200, 0)
                else:
                    skeleton_color = (0, 0, 255) # Red - Wrong/Detected but not target
                    keypoint_fill = (0, 0, 255)
                    keypoint_border = (0, 0, 180)
            else:
                skeleton_color = (0, 0, 255)  # Red - Waiting (Zoning/Countdown)
                keypoint_fill = (0, 0, 255)   
                keypoint_border = (0, 0, 180) 
            
            connections = []
            
            # Use different connections based on keypoint format
            if len(landmarks_abs) == 17:
                # YOLO-Pose (17 keypoints - COCO Format)
                # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LSh, 6:RSh, 7:LElb, 8:RElb, 9:LWri, 10:RWri, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnk, 16:RAnk
                connections = [
                    (5, 7), (7, 9),      # Left arm
                    (6, 8), (8, 10),     # Right arm
                    (5, 6),              # Shoulders
                    (5, 11), (6, 12),    # Torso
                    (11, 12),            # Hips
                    (11, 13), (13, 15),  # Left leg
                    (12, 14), (14, 16)   # Right leg
                ]
            else:
                # MediaPipe (33 keypoints)
                connections = [
                    (11, 13), (13, 15),  # Left arm
                    (12, 14), (14, 16),  # Right arm
                    (11, 12),            # Shoulders
                    (11, 23), (12, 24),  # Torso
                    (23, 24),            # Hips
                    (23, 25), (25, 27),  # Left leg
                    (24, 26), (26, 28),  # Right leg
                ]
            
            for connection in connections:
                if connection[0] < len(landmarks_abs) and connection[1] < len(landmarks_abs):
                    pt1 = landmarks_abs[connection[0]]
                    pt2 = landmarks_abs[connection[1]]
                    
                    # Skip if any point is invalid (undetected keypoint)
                    if pt1[0] < 0 or pt1[1] < 0 or pt2[0] < 0 or pt2[1] < 0:
                        continue
                    if (pt1[0] <= 1 and pt1[1] <= 1) or (pt2[0] <= 1 and pt2[1] <= 1):
                        continue
                        
                    # landmarks_abs contains (x, y, z) tuples
                    cv2.line(frame, (pt1[0], pt1[1]), 
                            (pt2[0], pt2[1]), skeleton_color, 2)
            
            # Draw keypoints
            for idx, landmark in enumerate(landmarks_abs):
                x, y = landmark[0], landmark[1]
                
                # Skip if point is invalid (sentinel -1 or at origin)
                if x < 0 or y < 0:
                    continue
                if x <= 1 and y <= 1:
                    continue
                    
                cv2.circle(frame, (x, y), 5, keypoint_fill, -1)
                cv2.circle(frame, (x, y), 6, keypoint_border, 2)
            
            # Draw stick if detected
            if 'stick_endpoints' in zone_data and zone_data['stick_endpoints']:
                grip_pt, tip_pt = zone_data['stick_endpoints']
                # Draw stick line (bright blue/cyan)
                cv2.line(frame, grip_pt, tip_pt, (255, 255, 0), 4)
                # Draw grip point (red circle)
                cv2.circle(frame, grip_pt, 8, (0, 0, 255), -1)
                cv2.circle(frame, grip_pt, 10, (255, 255, 255), 2)
                # Draw tip point (blue circle)
                cv2.circle(frame, tip_pt, 8, (255, 0, 0), -1)
                cv2.circle(frame, tip_pt, 10, (255, 255, 255), 2)

    def analyze_zones(self, frame):
        """Analyze each user zone using PoseAnalyzer with GCN"""
        if self.pose_analyzer is None:
            return {}
        
        h, w, _ = frame.shape
        col_w = w // self.num_users
        zone_results = {}
        
        for i in range(self.num_users):
            x_start = i * col_w
            x_end = (i + 1) * col_w
            zone_frame = frame[:, x_start:x_end].copy()
            
            # Set viewpoint for this user's GCN model
            # Direct mapping: models are trained with mirror augmentation
            # Models are now mirror-invariant and work with mirrored frames directly
            vp_raw = self.user_configs[i]['viewpoint']
            viewpoint_ui = vp_raw.get() if hasattr(vp_raw, 'get') else vp_raw
            viewpoint_mapping = {
                "Front": "front",
                "Right Side": "right",
                "Left Side": "left"
            }
            viewpoint = viewpoint_mapping.get(viewpoint_ui, viewpoint_ui).lower()
            
            if self.pose_analyzer.gcn_engine:
                self.pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            
            # NEW: Models are mirror-invariant, trained with 66% flipped augmentation
            # Send mirrored display frame directly to model without preprocessing
            # No flip needed - models understand poses regardless of mirror orientation
            
            # Analyze the zone with mirrored frame directly
            try:
                # MODE: snapshot - Lesson pose classification (no temporal smoothing)
                # Use MediaPipe for accurate snapshot classification
                results = self.pose_analyzer.process_frame(zone_frame, skip_ml_inference=False, mode='snapshot')
                
                if results:
                    # process_frame returns a list of person results
                    if isinstance(results, list) and len(results) > 0:
                        # Get the first person detected in this zone
                        person_data = results[0]
                        predicted_class = person_data.get('predicted_class', 'N/A')
                        confidence = person_data.get('confidence', 0.0)
                        
                        # Removed neutral_stance filter per user request
                        # if predicted_class.lower() == 'neutral_stance':
                        #     predicted_class = 'No Technique Detected'
                        #     confidence = 0.0
                        
                        # MIRROR FIX: The camera frame is cv2.flip(frame, 1) for
                        # display, so the GCN sees a mirrored pose — left↔right
                        # are swapped vs. the user's real-world body.  Swap the
                        # class label so it matches the user's actual perspective.
                        if 'left_' in predicted_class:
                            predicted_class = predicted_class.replace('left_', 'right_')
                        elif 'right_' in predicted_class:
                            predicted_class = predicted_class.replace('right_', 'left_')
                        
                        zone_results[i] = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'landmarks': person_data.get('landmarks'),
                            'landmarks_absolute': person_data.get('landmarks_absolute'),
                            'world_landmarks': person_data.get('world_landmarks'),
                            'live_angles': person_data.get('live_angles'),
                            'global_features': person_data.get('global_features'),
                            'stick_endpoints': person_data.get('stick_endpoints'),
                            'stick_detected': person_data.get('stick_endpoints') is not None,
                            'frame_w': person_data.get('frame_w', zone_frame.shape[1]),
                            'frame_h': person_data.get('frame_h', zone_frame.shape[0]),
                        }
                    elif isinstance(results, dict):
                        # Fallback for dict format (shouldn't happen but handle it)
                        for person_id, person_data in results.items():
                            predicted_class = person_data.get('predicted_class', 'N/A')
                            confidence = person_data.get('confidence', 0.0)
                            
                            # Filter out neutral_stance predictions
                            if predicted_class.lower() == 'neutral_stance':
                                predicted_class = 'No Technique Detected'
                                confidence = 0.0
                            
                            # MIRROR FIX: same left↔right swap as primary path
                            if 'left_' in predicted_class:
                                predicted_class = predicted_class.replace('left_', 'right_')
                            elif 'right_' in predicted_class:
                                predicted_class = predicted_class.replace('right_', 'left_')
                            
                            zone_results[i] = {
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'landmarks': person_data.get('landmarks'),
                                'landmarks_absolute': person_data.get('landmarks_absolute'),
                                'world_landmarks': person_data.get('world_landmarks'),
                                'live_angles': person_data.get('live_angles'),
                                'global_features': person_data.get('global_features'),
                                'stick_endpoints': person_data.get('stick_endpoints'),
                                'stick_detected': person_data.get('stick_endpoints') is not None,
                                'frame_w': person_data.get('frame_w', zone_frame.shape[1]),
                                'frame_h': person_data.get('frame_h', zone_frame.shape[0]),
                            }
                            break  # Only use first person in zone
                    else:
                        print(f"[Kiosk] Warning: Unexpected results type from pose_analyzer: {type(results)}")
            except Exception as e:
                print(f"[Kiosk] Error analyzing zone {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # DEMO OVERRIDE: Force specific outcomes for test images
        if DEMO_MODE and getattr(self, '_demo_image_name', None):
            if self._demo_image_name in DEMO_IMAGE_RESULTS:
                override = DEMO_IMAGE_RESULTS[self._demo_image_name]
                if 0 in zone_results:
                    # Force pose and confidence only
                    # DO NOT adjust landmarks/stick - YOLO already gives full-frame coordinates
                    # The portrait image is centered in the frame, detections are already correct
                    zone_results[0]['predicted_class'] = override['pose']
                    zone_results[0]['confidence'] = override['confidence']
                    # Store forced feedback for use in show_feedback
                    self._demo_forced_feedback = override.get('force_feedback')
                    self._demo_force_excellent = override.get('force_excellent', False)
                    print(f"[DEMO] Applied override for {self._demo_image_name}: confidence={override['confidence']}, pose={override['pose']}, stick={'detected' if zone_results[0].get('stick_endpoints') else 'not detected'}")
        
        return zone_results

    def update_feed(self):
        if not self.running: return
        self.frame_counter += 1
        
        VIDEO_ACTIVE_STATES = [AppState.ZONING, AppState.COUNTDOWN, AppState.SNAPSHOT, AppState.FEEDBACK, AppState.PAUSED]
        
        if self.app_state not in VIDEO_ACTIVE_STATES:
            if self.image_item: self.video_canvas.itemconfig(self.image_item, state="hidden")
            self.after(30, self.update_feed)
            return

        # Frame Capture
        if self.app_state == AppState.FEEDBACK and self.frozen_frame is not None:
            frame = self.frozen_frame.copy()
        elif self.app_state == AppState.PAUSED and self.frozen_frame is not None:
             if self.cap:
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1) if ret else np.zeros((720,1280,3),np.uint8)
        else:
            if DEMO_MODE:
                # TEMPORARY: Demo mode - load image instead of camera
                # 9:16 portrait aspect ratio with black letterbox bars
                try:
                    frame = cv2.imread(DEMO_IMAGE_PATH)
                    if frame is None:
                        print(f"[DEMO] Could not load image: {DEMO_IMAGE_PATH}")
                        frame = np.zeros((720, 1280, 3), np.uint8)
                    else:
                        # Portrait 9:16 aspect ratio
                        target_height = 720
                        target_width = int(target_height * 9 / 16)  # 9:16 ratio = 405px width
                        frame = cv2.resize(frame, (target_width, target_height))
                        # Pad with black bars to reach 1280x720, centered
                        padded = np.zeros((720, 1280, 3), dtype=np.uint8)
                        x_offset = (1280 - target_width) // 2
                        padded[:, x_offset:x_offset+target_width] = frame
                        frame = padded
                        # Store x_offset for stick positioning
                        self._demo_x_offset = x_offset
                        self._demo_frame_width = target_width
                    frame = cv2.flip(frame, 1)
                    self.current_frame = frame.copy()
                    self._demo_image_name = os.path.basename(DEMO_IMAGE_PATH)
                except Exception as e:
                    print(f"[DEMO] Error loading image: {e}")
                    frame = np.zeros((720, 1280, 3), np.uint8)
                    self.current_frame = frame.copy()
                    self._demo_x_offset = 0
                    self._demo_frame_width = 1280
            elif self.cap:
                ret, frame = self.cap.read()
                if not ret: frame = np.zeros((720, 1280, 3), np.uint8)
                else: frame = cv2.flip(frame, 1)
                # Store CLEAN frame before any drawing operations
                self.current_frame = frame.copy()
            else:
                 frame = np.zeros((720, 1280, 3), np.uint8)
                 self.current_frame = frame.copy()


        # Draw Zoning
        if self.app_state in [AppState.ZONING, AppState.COUNTDOWN, AppState.SNAPSHOT]:
            h, w, _ = frame.shape
            cols = self.num_users
            
            # Draw Separators (Lines)
            if cols > 1:
                col_w = w // cols
                for i in range(1, cols):
                    self.draw_vertical_separator(frame, i * col_w, h)
            
            # Show positioning status during ZONING
            if self.app_state == AppState.ZONING:
                elapsed = time.time() - self.zoning_start_time
                remaining = max(0, 10 - int(elapsed))
                
                status_text = f"Position yourself properly - {remaining}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
                
                text_x = (w - text_w) // 2
                text_y = 60
                
                # Semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (text_x - 20, text_y - text_h - 10), 
                            (text_x + text_w + 20, text_y + 10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # White text
                cv2.putText(frame, status_text, (text_x, text_y), font, font_scale, 
                           (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Draw Minimal User IDs (only for first 3 seconds in ZONING state)
            if self.app_state == AppState.ZONING and self.show_user_names and (time.time() - self.names_shown_time) < 3.0:
                for i in range(cols):
                    col_w = w // cols
                    cx = (i * col_w) + (col_w // 2)
                    
                    # Name
                    user_name = self.user_configs[i]['user']['name'] if self.user_configs[i]['user'] else f"P{i+1}"
                    
                    # Small Glass Badge at bottom
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), _ = cv2.getTextSize(user_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    tx = int(cx - (text_w / 2))
                    ty = int(h - 50)
                    
                    # Black Pill Background
                    pad = 10
                    cv2.rectangle(frame, (tx-pad, ty-text_h-pad), (tx+text_w+pad, ty+pad), (0,0,0), -1)
                    # Text
                    cv2.putText(frame, user_name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            elif self.app_state == AppState.ZONING and self.show_user_names and (time.time() - self.names_shown_time) >= 3.0:
                self.show_user_names = False
        
        # SAFEGUARD: Flag to prevent double skeleton drawings on the same frame
        skeletons_drawn_this_frame = False
        
        # Real-time keypoint drawing during ZONING, COUNTDOWN only (NOT SNAPSHOT to avoid double-drawing)
        if self.app_state in [AppState.ZONING, AppState.COUNTDOWN] and self.pose_analyzer and not skeletons_drawn_this_frame:
            h, w, _ = frame.shape
            col_w = w // self.num_users
            
            # Quick pose analysis for drawing (no ML inference needed)
            realtime_results = {}
            for i in range(self.num_users):
                x_start = i * col_w
                x_end = (i + 1) * col_w
                zone_frame = frame[:, x_start:x_end].copy()
                
                try:
                    # MODE: countdown - Countdown visualization (uses video mode for temporal smoothing)
                    # Run pose detection ONLY (disable stick detection for speed/clarity)
                    # Use YOLO-Pose for fast countdown visualization
                    results = self.pose_analyzer.process_frame(zone_frame, skip_ml_inference=True, skip_stick_detection=True, mode='countdown')
                    
                    # Update cache for check_zones_and_countdown
                    if results and len(results) > 0:
                       self.realtime_pose_cache[i] = results[0]
                    else:
                       self.realtime_pose_cache[i] = None
                       
                    if results and len(results) > 0:
                        person_data = results[0]
                        if 'landmarks_absolute' in person_data and person_data['landmarks_absolute']:
                            # Adjust landmarks for zone offset
                            adjusted_landmarks = [(x + x_start, y, z) for x, y, z in person_data['landmarks_absolute']]
                            realtime_results[i] = {
                                'landmarks_absolute': adjusted_landmarks
                            }
                            
                            # Add stick endpoints if detected (already in frame coordinates)
                            if 'stick_endpoints' in person_data and person_data['stick_endpoints']:
                                grip_pt, tip_pt = person_data['stick_endpoints']
                                # Adjust stick endpoints for zone offset
                                adjusted_grip = (grip_pt[0] + x_start, grip_pt[1])
                                adjusted_tip = (tip_pt[0] + x_start, tip_pt[1])
                                realtime_results[i]['stick_endpoints'] = (adjusted_grip, adjusted_tip)
                except:
                    pass
            
            if realtime_results:
                # Red keypoints during real-time (before prediction)
                try:
                    self.draw_pose_keypoints(frame, realtime_results, col_w, self.num_users, prediction_ready=False)
                except Exception as e:
                    print(f"Error drawing realtime keypoints: {e}")
        
        # Draw keypoints during FEEDBACK state (on frozen frame with analysis results)
        if self.app_state == AppState.FEEDBACK and hasattr(self, 'analysis_results') and self.analysis_results:
            h, w, _ = frame.shape
            col_w = w // self.num_users
            
            # Build complete results with per-zone success flags
            feedback_results = {}
            for i in range(self.num_users):
                if i in self.analysis_results:
                    try:
                        predicted = self.analysis_results[i].get('predicted_class', 'N/A')
                        conf = self.analysis_results[i].get('confidence', 0.0)

                        # Pure recognition mode - confidence-based coloring only
                        user_conf = self.user_configs[i]
                        vp_raw = user_conf['viewpoint']
                        viewpoint_ui = vp_raw.get() if hasattr(vp_raw, 'get') else vp_raw
                        viewpoint_mapping = {
                            "Front": "front",
                            "Right Side": "right",
                            "Left Side": "left"
                        }
                        viewpoint = viewpoint_mapping.get(viewpoint_ui, "front").lower()

                        gcn_config = self.pose_analyzer.gcn_engine.config if (self.pose_analyzer and self.pose_analyzer.gcn_engine) else {}
                        confidence_threshold = gcn_config.get('models', {}).get(viewpoint, {}).get('confidence_threshold', 0.55)

                        # Determine quality based purely on confidence
                        pose_detected = (predicted != 'N/A') and (predicted.lower() != 'no technique detected') and (conf > 0)
                        high_confidence = (conf >= 0.60)
                        good_confidence = (conf >= 0.40)

                        # Store result with confidence-based status for skeleton coloring
                        result_copy = self.analysis_results[i].copy()

                        # In guided lesson mode, also detect wrong technique → orange
                        target_key = self.current_lesson.get('key') if self.current_lesson else None
                        wrong_technique = (self.current_lesson is not None
                                          and pose_detected
                                          and target_key is not None
                                          and predicted != target_key)

                        # Adjust landmarks from zone-local to full-frame coordinates
                        x_start = i * col_w
                        if 'landmarks_absolute' in result_copy and result_copy['landmarks_absolute']:
                            result_copy['landmarks_absolute'] = [
                                (x + x_start, y, z) for x, y, z in result_copy['landmarks_absolute']
                            ]
                        # Adjust stick endpoints from zone-local to full-frame coordinates
                        if 'stick_endpoints' in result_copy and result_copy['stick_endpoints']:
                            grip_pt, tip_pt = result_copy['stick_endpoints']
                            result_copy['stick_endpoints'] = (
                                (grip_pt[0] + x_start, grip_pt[1]),
                                (tip_pt[0] + x_start, tip_pt[1])
                            )

                        if wrong_technique:
                            result_copy['status'] = 'wrong'   # Orange - wrong technique in lesson
                            result_copy['is_correct'] = False
                        elif not pose_detected:
                            result_copy['status'] = 'bad'     # Red - No detection
                            result_copy['is_correct'] = False
                        elif high_confidence:
                            result_copy['status'] = 'perfect' # Green - Excellent
                            result_copy['is_correct'] = True
                        elif good_confidence:
                            result_copy['status'] = 'good'    # Lime - Good
                            result_copy['is_correct'] = True
                        else:
                            result_copy['status'] = 'bad'     # Red - Fair/Low confidence
                            result_copy['is_correct'] = False

                        feedback_results[i] = result_copy
                    except Exception as e:
                        print(f"Error preparing feedback for user {i}: {e}")
            
            # Draw all skeletons in a single call with individual colors
            if feedback_results:
                try:
                    self.draw_pose_keypoints(frame, feedback_results, col_w, self.num_users, 
                                            prediction_ready=True, use_individual_colors=True)
                    skeletons_drawn_this_frame = True  # Mark that we've drawn skeletons
                except Exception as e:
                    print(f"Error drawing feedback keypoints: {e}")

        # Render
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        screen_ratio = self.screen_width / self.screen_height
        img_ratio = img.width / img.height
        
        if screen_ratio > img_ratio:
            new_height = self.screen_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = self.screen_width
            new_height = int(new_width / img_ratio)
            
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        final_img = Image.new("RGB", (self.screen_width, self.screen_height), (0, 0, 0))
        x_offset = (self.screen_width - new_width) // 2
        y_offset = (self.screen_height - new_height) // 2
        final_img.paste(img, (x_offset, y_offset))
        
        # Store display rect so feedback text can align with the actual video area
        self.video_x_offset = x_offset
        self.video_y_offset = y_offset
        self.video_display_width = new_width
        self.video_display_height = new_height
        
        if self.app_state == AppState.PAUSED:
            overlay = Image.new("RGBA", final_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(0,0), final_img.size], fill=(0, 0, 0, 150))
            final_img = final_img.convert("RGBA")
            final_img = Image.alpha_composite(final_img, overlay)
            final_img = final_img.convert("RGB")

        imgtk = ImageTk.PhotoImage(image=final_img)
        
        if self.image_item is None:
            self.image_item = self.video_canvas.create_image(0, 0, image=imgtk, anchor="nw", tags="video")
        else:
            self.video_canvas.itemconfig(self.image_item, image=imgtk, state="normal")
            self.video_canvas.tag_lower(self.image_item) 
            
        self.video_canvas.imgtk = imgtk 
        self.after(30, self.update_feed)

if __name__ == "__main__":
    try:
        app = KioskApp()
        app.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("CRITICAL ERROR: The application crashed.")
        print(f"Error: {e}")
        print("="*60 + "\n")
        input("Press Enter to exit...")
