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

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
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
        self.timer_text_id = None
        
        # Start Loop
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

    def add_text(self, x, y, text, font=FONT_MAIN, fill=COLOR_TEXT, anchor="center"):
        # Shadow
        if fill == "white" or fill == COLOR_TEXT_WHITE:
             self.video_canvas.create_text(x+2, y+2, text=text, font=font, fill="black", anchor=anchor)
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
        
        self.add_text(cx, cy - 80, "TuroArnis", font=("Inter", 96, "bold"), fill=COLOR_TEXT)
        self.add_text(cx, cy + 40, "Interactive Kiosk System", font=("Inter", 32), fill=COLOR_TEXT)
        
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

    def start_config(self, n):
        self.num_users = n
        self.user_configs = []
        for i in range(n):
            self.user_configs.append({
                'user': None,
                'viewpoint': ctk.StringVar(value="Front"),
                'form': ctk.StringVar(value="Pugay"),
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
            ctk.CTkSegmentedButton(card, values=["Front", "Left", "Right"], variable=self.user_configs[i]['viewpoint'], 
                                  font=("Inter", 16), fg_color="#ecf0f1", selected_color=COLOR_ACCENT, selected_hover_color=COLOR_ACCENT_HOVER, text_color="black").pack(pady=(5, 20), padx=20, fill="x")
            
            ctk.CTkLabel(card, text="Target Move", font=("Inter", 18, "bold"), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
            ctk.CTkOptionMenu(card, variable=self.user_configs[i]['form'], 
                             values=["Pugay", "Forward Stance", "Back Stance", "Left Temple Block", "Right Temple Block"],
                             font=("Inter", 16), height=40,
                             fg_color=COLOR_ACCENT, button_color=COLOR_ACCENT, button_hover_color=COLOR_ACCENT_HOVER, text_color="white").pack(pady=5, padx=20, fill="x")
            
            self.add_widget(cx, cy, card)

        btn = ctk.CTkButton(self.video_canvas, text="LOCK IN [ENTER]", font=("Inter", 32, "bold"),
                           fg_color=COLOR_SUCCESS, hover_color="#27ae60", height=90, width=400, corner_radius=45,
                           command=self.start_zoning_check)
        self.add_widget(self.screen_width//2, self.screen_height - 100, btn)

    def open_user_select(self, slot_index):
        dialog = UserManagementDialog(self, self.db)
        self.wait_window(dialog.dialog)
        
        selected_user = dialog.selected_user
        if selected_user:
            self.user_configs[slot_index]['user'] = selected_user
            self.show_config_screen()

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
        self.add_text(self.screen_width//2, 120, "GET INTO YOUR ZONE", font=("Inter", 64, "bold"), fill="white")
        self.after(3000, self.start_countdown)

    def start_countdown(self):
        self.kiosk_state = KioskState.COUNTDOWN
        self.clear_ui()
        self.countdown_timer = 5
        
        cx, cy = self.screen_width//2, self.screen_height//2
        self.count_text_id = self.video_canvas.create_text(cx, cy, text="5", font=("Inter", 200, "bold"), fill=COLOR_WARNING)
        self.canvas_items.append(self.count_text_id)
        
        self.update_countdown()

    def update_countdown(self):
        if self.kiosk_state != KioskState.COUNTDOWN: return

        if self.countdown_timer > 0:
            self.video_canvas.itemconfig(self.count_text_id, text=str(self.countdown_timer))
            self.countdown_timer -= 1
            self.after(1000, self.update_countdown)
        else:
            self.video_canvas.itemconfig(self.count_text_id, text="SNAP!", fill=COLOR_SUCCESS)
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
        self.feedback_timer = 10
        
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
                'Pugay': 'neutral_stance',
                'Forward Stance': 'forward_stance_correct',
                'Left Temple Block': 'left_temple_block_correct',
                'Right Temple Block': 'right_temple_block_correct',
            }
            
            expected_class = class_name_mapping.get(target_pose, 'neutral_stance')
            is_correct = (predicted_class == expected_class) and (confidence > 0.6)
            
            color = COLOR_SUCCESS if is_correct else "#f1c40f"
            
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
            self.add_text(cx, self.screen_height - 200, f"{score}%", font=("Inter", 80, "bold"), fill=color)
            
            # Show feedback message
            if predicted_class == 'N/A':
                feedback_msg = "No pose detected"
            elif is_correct:
                feedback_msg = "Perfect form!"
            else:
                feedback_msg = f"Detected: {predicted_class.replace('_', ' ').title()}"
            
            self.add_text(cx, self.screen_height - 130, feedback_msg, font=("Inter", 18), fill="white")

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
        self.add_text(cx, 80, "Session Results", font=("Inter", 64, "bold"), fill=COLOR_TEXT)
        
        container = ctk.CTkFrame(self.video_canvas, fg_color="transparent")
        self.add_widget(cx, 320, container)
        
        for i in range(self.num_users):
            user = self.user_configs[i]['user']
            
            card = ctk.CTkFrame(container, fg_color="white", width=300, height=400, corner_radius=20)
            card.pack(side="left", padx=30)
            card.pack_propagate(False)
            
            ctk.CTkLabel(card, text=user['name'], font=("Inter", 28, "bold"), text_color=COLOR_TEXT).pack(pady=(40, 20))
            ctk.CTkLabel(card, text="Data Saved", font=("Inter", 20), text_color=COLOR_SUCCESS).pack(pady=20)
            
            btn_details = ctk.CTkButton(card, text="VIEW HISTORY", font=FONT_BOLD,
                                       fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, height=60, width=200, corner_radius=15,
                                       command=lambda u=user: self.open_full_results(u))
            btn_details.pack(pady=30)
            
        btn_restart = ctk.CTkButton(self.video_canvas, text="NEW SESSION", font=("Inter", 28, "bold"),
                                   fg_color=COLOR_SUCCESS, height=80, width=350, corner_radius=40,
                                   command=self.show_splash)
        self.add_widget(cx, self.screen_height - 150, btn_restart)

    def open_full_results(self, user):
        rw = ResultsWindow(self, self.db, user)
    
    def update_feedback_timer(self):
        \"\"\"Update the feedback timer countdown\"\"\"
        if self.kiosk_state != KioskState.FEEDBACK:
            return
        
        if self.feedback_timer > 0:
            self.video_canvas.itemconfig(self.timer_text_id, text=f\"Next in {self.feedback_timer}...\")
            self.feedback_timer -= 1
            self.after(1000, self.update_feedback_timer)
        else:
            # Restart the countdown for next attempt
            self.start_countdown()

    # --- VIDEO ENGINE ---

    def draw_vertical_separator(self, frame, x, h):
        cv2.line(frame, (x, 50), (x, h-50), (255, 255, 255), 2)

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
            viewpoint = self.user_configs[i]['viewpoint'].get().lower()
            if self.pose_analyzer.gcn_engine:
                self.pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            
            # Analyze the zone
            try:
                results = self.pose_analyzer.process_frame(zone_frame, skip_ml_inference=False)
                
                if results:
                    # Get the first person detected in this zone
                    for person_id, person_data in results.items():
                        zone_results[i] = {
                            'predicted_class': person_data.get('predicted_class', 'N/A'),
                            'confidence': person_data.get('confidence', 0.0),
                            'landmarks': person_data.get('landmarks'),
                            'stick_detected': person_data.get('stick_endpoints') is not None
                        }
                        break  # Only use first person in zone
            except Exception as e:
                print(f"[Kiosk] Error analyzing zone {i}: {e}")
        
        return zone_results

    def update_feed(self):
        if not self.running: return
        
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
                self.current_frame = frame
            else:
                 frame = np.zeros((720, 1280, 3), np.uint8)

        # Draw Zoning
        if self.kiosk_state in [KioskState.ZONING, KioskState.COUNTDOWN, KioskState.SNAPSHOT]:
            h, w, _ = frame.shape
            cols = self.num_users
            
            # Draw Separators (Lines)
            if cols > 1:
                col_w = w // cols
                for i in range(1, cols):
                    self.draw_vertical_separator(frame, i * col_w, h)
            
            # Draw Minimal User IDs
            for i in range(cols):
                col_w = w // cols
                cx = (i * col_w) + (col_w // 2)
                
                # Name
                user_name = self.user_configs[i]['user']['name'] if self.user_configs[i]['user'] else f"P{i+1}"
                
                # Small Glass Badge at bottom
                # Text size
                font_scale = 0.7
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(user_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                tx = int(cx - (text_w / 2))
                ty = int(h - 50)
                
                # Black Pill Background
                pad = 10
                cv2.rectangle(frame, (tx-pad, ty-text_h-pad), (tx+text_w+pad, ty+pad), (0,0,0), -1)
                # Text
                cv2.putText(frame, user_name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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
    app = KioskApp()
    app.mainloop()
