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

# Viewpoint-specific confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'front': 0.35,  
    'left': 0.35,   
    'right': 0.35  
}

class KioskState:
    SPLASH = "splash"
    USER_COUNT = "user_count"
    CONFIG = "config"
    ZONING = "zoning"
    COUNTDOWN = "countdown"
    SNAPSHOT = "snapshot"
    FEEDBACK = "feedback"
    PAUSED = "paused"
    RESULTS = "results"

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
        
        # Bindings
        self.bind("<Escape>", lambda e: self.close_app())
        self.bind("<space>", self.on_spacebar)
        self.bind("<Return>", self.on_enter)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # State
        self.kiosk_state = KioskState.SPLASH
        self.previous_state = None
        self.num_users = 0
        self.user_configs = [] 
        self.feedback_timer = 0
        self.countdown_timer = 0
        self.analysis_results = {}  # Store pose analysis per user zone
        
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
        if self.kiosk_state == KioskState.PAUSED:
            self.resume_session()
        elif self.kiosk_state in [KioskState.ZONING, KioskState.COUNTDOWN, KioskState.SNAPSHOT, KioskState.FEEDBACK]:
            self.show_pause_menu()
            
    def on_enter(self, event):
        if self.kiosk_state == KioskState.CONFIG:
            self.start_zoning_check()

    # --- MENU ACTIONS ---
    def show_pause_menu(self):
        self.previous_state = self.kiosk_state
        self.kiosk_state = KioskState.PAUSED
        self.clear_ui()
        
        cx, cy = self.screen_width // 2, self.screen_height // 2
        
        self.add_text(cx, cy - 150, "SESSION PAUSED", font=("Inter", 64, "bold"), fill="white")
        
        btn_resume = ctk.CTkButton(self.video_canvas, text="RESUME", font=FONT_BOLD,
                                  fg_color=COLOR_SUCCESS, width=300, height=70, corner_radius=15,
                                  command=self.resume_session)
        self.add_widget(cx, cy - 30, btn_resume)

        btn_settings = ctk.CTkButton(self.video_canvas, text="RESTART SETUP", font=FONT_BOLD,
                                    fg_color="#f39c12", width=300, height=70, corner_radius=15,
                                    command=lambda: self.show_user_count()) 
        self.add_widget(cx, cy + 60, btn_settings)

        btn_quit = ctk.CTkButton(self.video_canvas, text="END SESSION", font=FONT_BOLD,
                                fg_color=COLOR_WARNING, width=300, height=70, corner_radius=15,
                                command=self.end_session_and_show_results)
        self.add_widget(cx, cy + 150, btn_quit)

    def resume_session(self):
        if self.previous_state:
            self.kiosk_state = self.previous_state
            if self.kiosk_state == KioskState.ZONING: self.start_zoning_check()
            elif self.kiosk_state == KioskState.FEEDBACK: self.show_feedback() 
            elif self.kiosk_state == KioskState.COUNTDOWN: self.start_countdown()

    def end_session_and_show_results(self):
        for config in self.user_configs:
            if config.get('session_id'):
                self.db.end_session(config['session_id'])
                config['session_id'] = None
        self.show_results_screen()

    # --- STATES ---

    def show_splash(self):
        self.kiosk_state = KioskState.SPLASH
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
                           command=self.show_user_count)
        self.add_widget(cx, cy + 180, btn)

    def show_user_count(self):
        self.kiosk_state = KioskState.USER_COUNT
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
        
        # Back Button
        btn_back = ctk.CTkButton(self.video_canvas, text="← BACK", font=("Inter", 24, "bold"),
                                fg_color="transparent", border_width=2, border_color=COLOR_TEXT, text_color=COLOR_TEXT,
                                hover_color="#ecf0f1", height=60, width=200, corner_radius=30,
                                command=self.show_splash)
        self.add_widget(cx, cy + 250, btn_back)

    def start_config(self, n):
        self.num_users = n
        self.user_configs = []
        for i in range(n):
            self.user_configs.append({
                'user': None,
                'viewpoint': ctk.StringVar(value="Front"),
                'form': ctk.StringVar(value="Left Temple Block"),
                'session_id': None
            })
        self.show_config_screen()

    def show_config_screen(self):
        self.kiosk_state = KioskState.CONFIG
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
            
            # Card (White on light blue)
            card = ctk.CTkFrame(self.video_canvas, fg_color="white", corner_radius=20, width=350, height=400)
            card.pack_propagate(False)
            
            current_name = self.user_configs[i]['user']['name'] if self.user_configs[i]['user'] else f"Guest {i+1}"
            btn_user = ctk.CTkButton(card, text=current_name, font=("Inter", 28, "bold"),
                                    fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, height=60, corner_radius=10,
                                    text_color="white",
                                    command=lambda idx=i: self.open_user_select(idx))
            btn_user.pack(pady=(30, 20), padx=20, fill="x")
            
            ctk.CTkLabel(card, text="Tap to change user", font=("Inter", 14), text_color="gray").pack(pady=(0, 20))
            
            ctk.CTkLabel(card, text="Viewpoint", font=("Inter", 18, "bold"), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
            ctk.CTkSegmentedButton(card, values=["Front", "Right Side", "Left Side"], variable=self.user_configs[i]['viewpoint'], 
                                  font=("Inter", 16), fg_color="#ecf0f1", selected_color=COLOR_ACCENT, selected_hover_color=COLOR_ACCENT_HOVER, text_color="black").pack(pady=(5, 20), padx=20, fill="x")
            
            ctk.CTkLabel(card, text="Target Move", font=("Inter", 18, "bold"), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
            ctk.CTkOptionMenu(card, variable=self.user_configs[i]['form'], 
                             values=[
                                 "Crown Thrust",
                                 "Solar Plexus Thrust",
                                 "Left Chest Thrust",
                                 "Right Chest Thrust",
                                 "Left Eye Thrust",
                                 "Right Eye Thrust",
                                 "Left Temple Block",
                                 "Right Temple Block",
                                 "Left Elbow Block",
                                 "Right Elbow Block",
                                 "Left Knee Block",
                                 "Right Knee Block"
                             ],
                             font=("Inter", 16), height=40,
                             fg_color=COLOR_ACCENT, button_color=COLOR_ACCENT, button_hover_color=COLOR_ACCENT_HOVER, text_color="white").pack(pady=5, padx=20, fill="x")
            
            self.add_widget(cx, cy, card)

        btn = ctk.CTkButton(self.video_canvas, text="LOCK IN [ENTER]", font=("Inter", 32, "bold"),
                           fg_color=COLOR_SUCCESS, hover_color="#27ae60", height=90, width=400, corner_radius=45,
                           command=self.start_zoning_check)
        self.add_widget(self.screen_width//2, self.screen_height - 100, btn)
        
        # Back Button
        btn_back = ctk.CTkButton(self.video_canvas, text="← BACK", font=("Inter", 24, "bold"),
                                fg_color="transparent", border_width=2, border_color="white", text_color="white",
                                hover_color="#ffffff", height=60, width=150, corner_radius=30,
                                command=self.show_user_count)
        self.add_widget(150, self.screen_height - 100, btn_back)

    def open_user_select(self, slot_index):
        dialog = UserManagementDialog(self, self.db)
        self.wait_window(dialog.dialog)
        
        selected_user = dialog.selected_user
        if selected_user:
            self.user_configs[slot_index]['user'] = selected_user
            self.show_config_screen()

    def restart_zoning(self):
        """Restart the zoning phase for a new repetition"""
        self.kiosk_state = KioskState.ZONING
        self.clear_ui()
        self.frozen_frame = None
        self.analysis_results = {} 
        self.show_user_names = True
        self.names_shown_time = time.time()
        self.zoning_start_time = time.time()
        self.after(200, self.check_zones_and_countdown)

    def start_zoning_check(self):
        for config in self.user_configs:
            user_id = config['user']['id'] if config['user'] else None
            if not user_id:
                guest_name = f"Guest_{int(time.time())}_{np.random.randint(100)}"
                user_id = self.db.create_user(guest_name)
                config['user'] = self.db.get_user_by_id(user_id)
            
            sid = self.db.start_session(user_id, target_pose=config['form'].get())
            config['session_id'] = sid
            
        self.kiosk_state = KioskState.ZONING
        self.clear_ui()
        self.show_user_names = True
        self.names_shown_time = time.time()
        # Track zoning start time for 10-second timeout
        self.zoning_start_time = time.time()
        # Check if users are in zones before starting countdown
        self.check_zones_and_countdown()
    
    def check_zones_and_countdown(self):
        """Validate that users are properly positioned in zones before starting countdown"""
        if self.kiosk_state != KioskState.ZONING:
            return
        
        # Check for 10-second timeout
        elapsed_time = time.time() - self.zoning_start_time
        if elapsed_time > 10.0:
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
                        results = self.pose_analyzer.process_frame(zone_frame, skip_ml_inference=True, skip_stick_detection=False)
                        if results and len(results) > 0:
                            person_data = results[0]

                    if person_data:
                        landmarks = person_data.get('landmarks_absolute')
                        
                        if landmarks and len(landmarks) >= 33:
                            # Check visibility and positioning
                            # Key landmarks: nose (0), shoulders (11,12), hips (23,24), ankles (27,28)
                            visible_count = 0
                            key_indices = [0, 11, 12, 23, 24, 27, 28]
                            
                            for idx in key_indices:
                                if idx < len(landmarks):
                                    x, y, z = landmarks[idx]
                                    # Check if landmark is within zone bounds (with margin)
                                    if 0 < x < zone_w and 0 < y < zone_h:
                                        visible_count += 1
                            
                            # Require at least 5 out of 7 key landmarks visible
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
        self.kiosk_state = KioskState.COUNTDOWN
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
        
        self.update_countdown()

    def update_countdown(self):
        if self.kiosk_state != KioskState.COUNTDOWN: return

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
        self.kiosk_state = KioskState.SNAPSHOT
        if self.current_frame is not None:
            self.frozen_frame = self.current_frame.copy()
            # Analyze poses in the snapshot
            if self.pose_analyzer:
                self.analysis_results = self.analyze_zones(self.frozen_frame)
        self.after(800, self.show_feedback)

    def show_feedback(self):
        self.kiosk_state = KioskState.FEEDBACK
        self.clear_ui()
        self.feedback_timer = 6
        
        col_w = self.screen_width // self.num_users
        for i in range(self.num_users):
            cx = (i * col_w) + (col_w // 2)
            config = self.user_configs[i]
            
            # Get GCN analysis results for this zone
            zone_result = self.analysis_results.get(i, {})
            predicted_class = zone_result.get('predicted_class', 'N/A')
            confidence = zone_result.get('confidence', 0.0)
            stick_detected = zone_result.get('stick_detected', False)
            
            # Convert confidence to score (0-100)
            score = int(confidence * 100)
            
            # Determine if pose matches target
            target_pose = config['form'].get()
            class_name_mapping = {
                'Crown Thrust': 'crown_thrust_correct',
                'Solar Plexus Thrust': 'solar_plexus_thrust_correct',
                'Left Chest Thrust': 'left_chest_thrust_correct',
                'Right Chest Thrust': 'right_chest_thrust_correct',
                'Left Eye Thrust': 'left_eye_thrust_correct',
                'Right Eye Thrust': 'right_eye_thrust_correct',
                'Left Temple Block': 'left_temple_block_correct',
                'Right Temple Block': 'right_temple_block_correct',
                'Left Elbow Block': 'left_elbow_block_correct',
                'Right Elbow Block': 'right_elbow_block_correct',
                'Left Knee Block': 'left_knee_block_correct',
                'Right Knee Block': 'right_knee_block_correct',
            }
            
            expected_class = class_name_mapping.get(target_pose)
            
            # Get viewpoint-specific confidence threshold
            # Direct mapping: models are now mirror-invariant
            viewpoint_ui = config['viewpoint'].get()
            viewpoint_mapping = {
                "Front": "front",
                "Right Side": "right",
                "Left Side": "left"
            }
            viewpoint = viewpoint_mapping.get(viewpoint_ui, "front").lower()
            confidence_threshold = CONFIDENCE_THRESHOLDS.get(viewpoint, 0.50)
            
            # TEMPORARY: Show green for ANY pose with high confidence (for testing)
            # This allows users to verify the system is working
            is_correct = (confidence > confidence_threshold) and (predicted_class != 'N/A') and (predicted_class.lower() != 'no technique detected')
            
            # Debug logging (Silenced for packaging)
            # print(f"[DEBUG] Zone {i}: predicted='{predicted_class}', expected='{expected_class}', conf={confidence:.2f}, threshold={confidence_threshold:.2f}, is_correct={is_correct}")
            
            color = COLOR_SUCCESS if is_correct else "#f1c40f"
            if predicted_class == 'N/A' or predicted_class.lower() == 'no technique detected' or confidence == 0:
                 color = "#e74c3c" # Red for failed detection
            
            if config['session_id']:
                self.db.save_performance(
                    session_id=config['session_id'],
                    user_id=config['user']['id'],
                    pose_detected=predicted_class,
                    confidence=confidence,
                    is_correct=is_correct,
                    stick_detected=stick_detected
                )
            
            self.add_text(cx, self.screen_height - 280, config['user']['name'], font=("Inter", 24, "bold"), fill="white")
            
            # Show Qualitative Score instead of Percentage
            score_text = ""
            score_color = color
            
            if confidence > 0:
                if is_correct:
                    if confidence >= 0.40:
                        score_text = "PERFECT!"
                        score_color = "#2ecc71" # Emerald Green
                    else:
                        score_text = "GOOD!"
                        score_color = "#bfff00" # Lime Green (Yellowish-Green)
                else:
                    score_text = "ADJUST"
                    score_color = "#e74c3c" # Red
                
                # Debug logging silenced for packaging
                # print(f"[FEEDBACK-DEBUG] Zone {i}: Conf={confidence:.3f}, Text='{score_text}', Color={score_color}, Correct={is_correct}")
                
                # Reduce font size for text (was 80 for number)
                self.add_text(cx, self.screen_height - 200, score_text, font=("Inter", 60, "bold"), fill=score_color)
            else:
                 self.add_text(cx, self.screen_height - 200, "--", font=("Inter", 80, "bold"), fill=color)
            
            # Extract data earlier so it's available for all branches
            viewpoint = config['viewpoint'].get().lower()
            live_angles = zone_result.get('live_angles')
            landmarks = zone_result.get('landmarks')

            # Show feedback message
            if predicted_class == 'N/A' or predicted_class == 'No Technique Detected':
                feedback_msg = "Pose not recognized"
                # Even if not recognized, check form against target
                if expected_class and live_angles:
                    tips = self.generate_form_feedback(expected_class, viewpoint, live_angles, landmarks)
                    if "Good form" not in tips and "Goal" not in tips:
                        feedback_msg += f"\n{tips}"
                    else:
                         # Provide hint if no specific bad deviation found but still not recognized
                         feedback_msg += f"\nGoal: {target_pose}"
                elif expected_class:
                    feedback_msg += f"\nGoal: {target_pose}"
            elif is_correct:
                feedback_msg = "Perfect form!"
            else:
                # Wrong technique detected - give deviation feedback
                if live_angles and expected_class:
                    tips = self.generate_form_feedback(expected_class, viewpoint, live_angles, landmarks)
                    # If tips returned "Good form" or just "Goal: X", but we are in this block, 
                    # it means our angles match the template BUT the classifier is confused.
                    if "Good form" in tips:
                         feedback_msg = f"Goal: {target_pose}\nAdjust your form"
                    elif "Goal" in tips:
                         feedback_msg = tips # Just show goal
                    else:
                        feedback_msg = tips # Show specific tips (e.g. "Extend Arm")
                else:
                    feedback_msg = f"Goal: {target_pose}\nAdjust your form"
            
            self.add_text(cx, self.screen_height - 130, feedback_msg, font=("Inter", 18), fill="white")
        
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

        self.timer_text_id = self.video_canvas.create_text(self.screen_width//2, 50, 
                                                          text=f"Next in {self.feedback_timer}...", 
                                                          font=("Inter", 28), fill="white")
        self.canvas_items.append(self.timer_text_id)
        
        btn = ctk.CTkButton(self.video_canvas, text="FINISH", width=150, height=50, 
                           fg_color=COLOR_WARNING, font=("Inter", 18, "bold"),
                           command=self.end_session_and_show_results)
        self.add_widget(self.screen_width - 120, 50, btn)
        
        self.update_feedback_timer()

    def show_results_screen(self):
        self.kiosk_state = KioskState.RESULTS
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
        if self.kiosk_state != KioskState.FEEDBACK:
            return
        
        if self.feedback_timer > 0:
            self.video_canvas.itemconfig(self.timer_text_id, text=f"Next in {self.feedback_timer}...")
            self.feedback_timer -= 1
            self.after(1000, self.update_feedback_timer)
        else:
            # Restart the countdown via zoning validation for proper resets
            self.restart_zoning()

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
                     skeleton_color = (0, 255, 0)  # Green - Perfect
                     keypoint_fill = (0, 255, 0)
                     keypoint_border = (0, 200, 0)
                elif status == 'good':
                     # Lime Green / Yellowish-Green (BGR: Blue=0, Green=255, Red=191)
                     skeleton_color = (0, 255, 191) 
                     keypoint_fill = (0, 255, 191)
                     keypoint_border = (0, 200, 150)
                elif is_correct: # Fallback for boolean True without status
                     skeleton_color = (0, 255, 0) 
                     keypoint_fill = (0, 255, 0)
                     keypoint_border = (0, 200, 0)
                else:
                     skeleton_color = (0, 0, 255)  # Red - Bad
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
                    
                    # Skip if any point is at (0,0) - undetected in YOLO-Pose
                    if (pt1[0] <= 1 and pt1[1] <= 1) or (pt2[0] <= 1 and pt2[1] <= 1):
                        continue
                        
                    # landmarks_abs contains (x, y, z) tuples
                    cv2.line(frame, (pt1[0], pt1[1]), 
                            (pt2[0], pt2[1]), skeleton_color, 2)
            
            # Draw keypoints
            for idx, landmark in enumerate(landmarks_abs):
                x, y = landmark[0], landmark[1]
                
                # Skip if point is at (0,0)
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
            viewpoint_ui = self.user_configs[i]['viewpoint'].get()
            viewpoint_mapping = {
                "Front": "front",
                "Right Side": "right",
                "Left Side": "left"
            }
            viewpoint = viewpoint_mapping.get(viewpoint_ui, "front").lower()
            
            if self.pose_analyzer.gcn_engine:
                self.pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            
            # NEW: Models are mirror-invariant, trained with 66% flipped augmentation
            # Send mirrored display frame directly to model without preprocessing
            # No flip needed - models understand poses regardless of mirror orientation
            
            # Analyze the zone with mirrored frame directly
            try:
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
                        
                        # NEW: No coordinate flipping needed!
                        # Models return coordinates in mirrored space, matching the display frame
                        # Landmarks and stick coordinates align directly with the user's view
                        
                        zone_results[i] = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'landmarks': person_data.get('landmarks'),
                            'landmarks_absolute': person_data.get('landmarks_absolute'),
                            'stick_endpoints': person_data.get('stick_endpoints'),
                            'stick_detected': person_data.get('stick_endpoints') is not None
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
                            
                            # NEW: No coordinate flipping needed!
                            # Models return coordinates in mirrored space, matching the display frame
                            
                            zone_results[i] = {
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'landmarks': person_data.get('landmarks'),
                                'landmarks_absolute': person_data.get('landmarks_absolute'),
                                'stick_endpoints': person_data.get('stick_endpoints'),
                                'stick_detected': person_data.get('stick_endpoints') is not None
                            }
                            break  # Only use first person in zone
                    else:
                        print(f"[Kiosk] Warning: Unexpected results type from pose_analyzer: {type(results)}")
            except Exception as e:
                print(f"[Kiosk] Error analyzing zone {i}: {e}")
                import traceback
                traceback.print_exc()
        
        return zone_results

    def update_feed(self):
        if not self.running: return
        self.frame_counter += 1
        
        VIDEO_ACTIVE_STATES = [KioskState.ZONING, KioskState.COUNTDOWN, KioskState.SNAPSHOT, KioskState.FEEDBACK, KioskState.PAUSED]
        
        if self.kiosk_state not in VIDEO_ACTIVE_STATES:
            if self.image_item: self.video_canvas.itemconfig(self.image_item, state="hidden")
            self.after(30, self.update_feed)
            return

        # Frame Capture
        if self.kiosk_state == KioskState.FEEDBACK and self.frozen_frame is not None:
            frame = self.frozen_frame.copy()
        elif self.kiosk_state == KioskState.PAUSED and self.frozen_frame is not None:
             if self.cap:
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1) if ret else np.zeros((720,1280,3),np.uint8)
        else:
            if self.cap:
                ret, frame = self.cap.read()
                if not ret: frame = np.zeros((720, 1280, 3), np.uint8)
                else: frame = cv2.flip(frame, 1)
                # Store CLEAN frame before any drawing operations
                self.current_frame = frame.copy()
            else:
                 frame = np.zeros((720, 1280, 3), np.uint8)
                 self.current_frame = frame.copy()


        # Draw Zoning
        if self.kiosk_state in [KioskState.ZONING, KioskState.COUNTDOWN, KioskState.SNAPSHOT]:
            h, w, _ = frame.shape
            cols = self.num_users
            
            # Draw Separators (Lines)
            if cols > 1:
                col_w = w // cols
                for i in range(1, cols):
                    self.draw_vertical_separator(frame, i * col_w, h)
            
            # Show positioning status during ZONING
            if self.kiosk_state == KioskState.ZONING:
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
            if self.kiosk_state == KioskState.ZONING and self.show_user_names and (time.time() - self.names_shown_time) < 3.0:
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
            elif self.kiosk_state == KioskState.ZONING and self.show_user_names and (time.time() - self.names_shown_time) >= 3.0:
                self.show_user_names = False
        
        # SAFEGUARD: Flag to prevent double skeleton drawings on the same frame
        skeletons_drawn_this_frame = False
        
        # Real-time keypoint drawing during ZONING, COUNTDOWN only (NOT SNAPSHOT to avoid double-drawing)
        if self.kiosk_state in [KioskState.ZONING, KioskState.COUNTDOWN] and self.pose_analyzer and not skeletons_drawn_this_frame:
            h, w, _ = frame.shape
            col_w = w // self.num_users
            
            # Quick pose analysis for drawing (no ML inference needed)
            realtime_results = {}
            for i in range(self.num_users):
                x_start = i * col_w
                x_end = (i + 1) * col_w
                zone_frame = frame[:, x_start:x_end].copy()
                
                try:
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
        if self.kiosk_state == KioskState.FEEDBACK and hasattr(self, 'analysis_results') and self.analysis_results:
            h, w, _ = frame.shape
            col_w = w // self.num_users
            
            # Build complete results with per-zone success flags
            feedback_results = {}
            for i in range(self.num_users):
                if i in self.analysis_results:
                    try:
                        predicted = self.analysis_results[i].get('predicted_class', 'N/A')
                        user_conf = self.user_configs[i]
                        target = user_conf['form'].get() if user_conf.get('form') else ""
                        
                        # Convert to internal name for matching
                        class_name_mapping = {
                            'Crown Thrust': 'crown_thrust_correct', 'Solar Plexus Thrust': 'solar_plexus_thrust_correct',
                            'Left Chest Thrust': 'left_chest_thrust_correct', 'Right Chest Thrust': 'right_chest_thrust_correct',
                            'Left Eye Thrust': 'left_eye_thrust_correct', 'Right Eye Thrust': 'right_eye_thrust_correct',
                            'Left Temple Block': 'left_temple_block_correct', 'Right Temple Block': 'right_temple_block_correct',
                            'Left Elbow Block': 'left_elbow_block_correct', 'Right Elbow Block': 'right_elbow_block_correct',
                            'Left Knee Block': 'left_knee_block_correct', 'Right Knee Block': 'right_knee_block_correct',
                        }
                        expected = class_name_mapping.get(target)
                        
                        # Get viewpoint-specific confidence threshold
                        # Direct mapping: models are now mirror-invariant
                        viewpoint_ui = user_conf['viewpoint'].get()
                        viewpoint_mapping = {
                            "Front": "front",
                            "Right Side": "right",
                            "Left Side": "left"
                        }
                        viewpoint = viewpoint_mapping.get(viewpoint_ui, "front").lower()
                        
                        # UPDATED THRESHOLDS: Lowered by 0.10 for more forgiving assessment
                        confidence_thresholds = {
                            "front": 0.35,  # Aggressively lowered
                            "left": 0.35,  
                            "right": 0.35 
                        }
                        confidence_threshold = confidence_thresholds.get(viewpoint, 0.35)
                        
                        conf = self.analysis_results[i].get('confidence', 0.0)
                        
                        # TEMPORARY: Show green for ANY pose with high confidence
                        is_correct = (conf > confidence_threshold) and (predicted != 'N/A') and (predicted.lower() != 'no technique detected')
                        
                        # Store result with success flag for this zone
                        result_copy = self.analysis_results[i].copy()
                        result_copy['is_correct'] = is_correct
                        
                        # Determine detailed status for coloring
                        if is_correct:
                            if conf >= 0.40:
                                result_copy['status'] = 'perfect' # Green
                            else:
                                result_copy['status'] = 'good'    # Yellow
                        else:
                            result_copy['status'] = 'bad'         # Red
                        
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
        
        if self.kiosk_state == KioskState.PAUSED:
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
