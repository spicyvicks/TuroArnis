import sys
import os
import cv2
import threading
import time
import re
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
import queue
import numpy as np
import mediapipe as mp

# Set CustomTkinter appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

#add project root to sys.path for dev execution
if not getattr(sys, 'frozen', False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.gui.results_window import ResultsWindow
from app.gui.user_dialog import show_user_dialog
from app.gui.toast import ToastNotification
from app.gui.loading_spinner import LoadingSpinner
# from app.gui.splash_screen import SplashScreen  # Integrated inline
# from app.gui.status_bar import StatusBar  # Moved to controls panel
from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.database.db_manager import DatabaseManager
from app.utils.resource_path import get_resource_path, get_app_data_path

class TuroArnisGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        #set app icon for taskbar
        icon_path = get_resource_path('app/assets/TA.ico')
        print(f"[DEBUG-ICON] Icon path resolved to: {icon_path}")
        print(f"[DEBUG-ICON] Icon exists: {os.path.exists(icon_path)}")
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
            print(f"[DEBUG-ICON] Icon set successfully")
        else:
            print(f"[DEBUG-ICON] Icon file not found at {icon_path}")
        
        self.window.update_idletasks()
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        
        # Create splash frame inside main window with pale blue background
        self.splash_frame = ctk.CTkFrame(self.window, fg_color="#74b9ff", corner_radius=0)
        self.splash_frame.pack(fill="both", expand=True)
        
        # Splash content
        splash_content = ctk.CTkFrame(self.splash_frame, fg_color="#74b9ff")
        splash_content.place(relx=0.5, rely=0.5, anchor="center")
        
        # Add logo image
        try:
            logo_path = get_resource_path('app/assets/TA.png')
            from PIL import Image
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((120, 120), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            ctk.CTkLabel(
                splash_content,
                image=self.logo_photo,
                text="",
                fg_color="#74b9ff"
            ).pack(pady=(0, 20))
        except Exception as e:
            print(f"[WARN] Could not load logo: {e}")
        
        ctk.CTkLabel(
            splash_content,
            text="TuroArnis",
            font=("Inter", 48, "bold"),
            fg_color="#74b9ff",
            text_color="black"
        ).pack(pady=(0, 5))
        
        ctk.CTkLabel(
            splash_content,
            text="Arnis Form Correction System",
            font=("Inter", 16),
            fg_color="#74b9ff",
            text_color="black"
        ).pack(pady=(0, 30))
        
        self.splash_progress = ctk.CTkProgressBar(
            splash_content,
            mode='indeterminate',
            width=300,
            progress_color="#3498db"
        )
        self.splash_progress.pack(pady=(0, 10))
        self.splash_progress.start()
        
        self.splash_status = ctk.CTkLabel(
            splash_content,
            text="Initializing...",
            font=("Inter", 14),
            fg_color="#74b9ff",
            text_color="black"
        )
        self.splash_status.pack()
        
        # Center and size the window to match main window size (80% of screen)
        width = int(self.screen_width * 0.8)
        height = int(self.screen_height * 0.8)
        x = (self.screen_width - width) // 2
        y = (self.screen_height - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.update()

        #app data directory for database (persists across updates)
        db_path = os.path.join(get_app_data_path(), 'turoarnis.db')
        self.db = DatabaseManager(db_path)
        print(f"[INFO] Database location: {db_path}")
        self.current_user = None
        self.current_session_id = None
        
        # Wait 3 seconds to show splash screen - update periodically for animation
        import time
        for i in range(30):  # 30 iterations of 0.1 seconds = 3 seconds
            time.sleep(0.1)
            self.window.update()  # Keep progress bar animating
        
        print("[DEBUG-INIT] Showing user selection dialog...")
        self.splash_status.config(text="Loading user management...")
        self.window.update()
        for i in range(10):  # 1 second with animation
            time.sleep(0.1)
            self.window.update()
        
        # Hide splash screen before showing user dialog
        self.splash_frame.pack_forget()
        self.window.update()
        
        self.show_user_selection()
        
        # Show splash screen again after user selection
        if self.current_user:
            self.splash_frame.pack(fill=BOTH, expand=YES)
            self.window.update()
        
        print(f"[DEBUG-INIT] After show_user_selection, current_user = {self.current_user}")
        
        if not self.current_user:
            print("[INFO] no user selected, exiting...")
            self.window.destroy()
            return

        print(f"[DEBUG-INIT] User validated: {self.current_user['name']}")
        print("[DEBUG-INIT] Initializing frame counters...")
        self.frame_counter = 0
        self.processing_interval = 1  #process every frame for smooth skeleton
        self.ml_inference_interval = 8  #run ml classification less frequently
        self.stick_detection_interval = 4  #run stick detection every 4th frame
        self.last_known_results = []
        self.last_ml_inference_frame = 0  #when we ran ml classifier
        self.last_stick_detection_frame = 0  #when we ran stick detector
        
        print("[DEBUG-INIT] Initializing state tracking...")
        #state tracking configuration
        self.MIN_STATE_FRAMES = 10  #reduced from 15 for faster response (0.1-0.3s)
        self.MAX_STATE_DURATION = 300  #timeout after ~3-10s depending on fps
        
        #state tracking variables
        self.last_pose_state = None
        self.state_frame_count = 0
        
        #landmark smoothing buffer (reduces jitter)
        self.landmark_smooth_buffer = {}  # Dictionary keyed by person_id
        self.smooth_window = 3  # Smooth over 3 frames
        self.last_person_bbox = {}  # Track bounding boxes to detect movement

        #initialize ux components
        print("[DEBUG-INIT] Initializing UX components...")
        self.toast = ToastNotification(self.window)

        print("[DEBUG-INIT] Loading stick detector model...")
        self.splash_status.config(text="Loading stick detector...")
        self.window.update()
        import time
        time.sleep(0.3)  # Pause to show status
        self.window.update()
        
        #use resource path for stick detector model
        stick_model_relative = 'runs/pose/arnis_stick_detector/weights/best.pt'
        stick_model_path = get_resource_path(stick_model_relative)
        print(f"[DEBUG-INIT] Stick model path: {stick_model_path}")
        
        print("[DEBUG-INIT] Initializing PoseAnalyzer...")
        self.splash_status.config(text="Loading AI models...")
        self.window.update()
        import time
        time.sleep(0.3)  # Pause to show status
        
        self.analyzer = PoseAnalyzer(
            detection_interval=self.processing_interval,
            stick_model_path=stick_model_path if os.path.exists(stick_model_path) else None,
            debug_stick=False
        )
        import time
        time.sleep(0.3)  # Pause to show status
        
        print("[DEBUG-INIT] Opening camera...")
        self.splash_status.config(text="Connecting to camera...")
        self.window.update()
        self.window.update()
        
        self.cap = cv2.VideoCapture(0)
        
        #check if camera opened successfully
        if not self.cap.isOpened():
            self.splash_progress.stop()
            self.splash_frame.destroy()
            self.toast.show("Camera not detected. Please check your camera connection.", "error", duration=5000)
            print("[ERROR] Camera failed to open")
        else:
            #get camera resolution
            cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] Camera opened: {cam_width}x{cam_height}")
            
            # Destroy splash screen before showing video interface
            self.splash_progress.stop()
            self.splash_frame.destroy()
        
        print("[DEBUG-INIT] Setting up GUI components...")
        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        
        # Create main container frame to avoid geometry manager conflicts
        main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        main_container.pack(fill="both", expand=True)
        
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=0)
        main_container.grid_columnconfigure(1, weight=1) 

        self.video_canvas = tk.Canvas(main_container, background='black', highlightthickness=0)
        self.video_canvas.grid(row=0, column=1, sticky="nsew")
        self.video_canvas.bind('<Configure>', self.on_canvas_resize)
        self.tk_image = None

        self.controls_panel = ctk.CTkFrame(main_container, width=250, corner_radius=0, fg_color="#f0f0f0")
        self.controls_panel.grid(row=0, column=0, sticky="nsew")
        self.controls_panel.grid_propagate(False) 
        
        ctk.CTkLabel(self.controls_panel, text="Controls", font=("Inter", 18, "bold"), text_color="#2c3e50").pack(pady=(10, 10), anchor="w", padx=15)

        user_frame = ctk.CTkFrame(self.controls_panel, corner_radius=10)
        user_frame.pack(fill="x", pady=5, padx=10)
        
        print(f"[DEBUG-INIT] Creating user label with name: {self.current_user['name']}")
        ctk.CTkLabel(user_frame, text=self.current_user['name'], font=("Inter", 16, "bold"), text_color="#27ae60").pack(anchor="w", padx=10)
        ctk.CTkLabel(user_frame, text=f"ID: {self.current_user['id']}", font=("Inter", 12), text_color="#95a5a6").pack(anchor="w", padx=10, pady=(0, 10))

        session_frame = ctk.CTkFrame(self.controls_panel, corner_radius=10)
        session_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(session_frame, text="Session", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        
        self.session_status_label = ctk.CTkLabel(session_frame, text="No active session", font=("Inter", 12), text_color="#f39c12")
        self.session_status_label.pack(anchor="w", pady=2, padx=10)
        
        session_btn_frame = ctk.CTkFrame(session_frame, fg_color="transparent")
        session_btn_frame.pack(fill="x", pady=5, padx=10)
        
        self.start_session_btn = ctk.CTkButton(session_btn_frame, text="Start", command=self.manual_start_session, fg_color="#27ae60", hover_color="#229954", width=100, corner_radius=20, font=("Inter", 14))
        self.start_session_btn.pack(side="left", padx=2)
        
        self.end_session_btn = ctk.CTkButton(session_btn_frame, text="End", command=self.end_session, fg_color="#e74c3c", hover_color="#c0392b", width=100, corner_radius=20, font=("Inter", 14), state="disabled")
        self.end_session_btn.pack(side="left", padx=2)
        
        # Separator
        ctk.CTkFrame(self.controls_panel, height=2, fg_color="#bdc3c7").pack(fill="x", pady=10, padx=15)
        
        self.practice_stances = {
            "Crown Thrust": "crown_thrust_correct", "Left Chest Thrust": "left_chest_thrust_correct",
            "Left Elbow Block": "left_elbow_block_correct", "Left Eye Thrust": "left_eye_thrust_correct",
            "Left Knee Block": "left_knee_block_correct", "Left Temple Block": "left_temple_block_correct",
            "Right Chest Thrust": "right_chest_thrust_correct", "Right Elbow Block": "right_elbow_block_correct",
            "Right Eye Thrust": "right_eye_thrust_correct", "Right Knee Block": "right_knee_block_correct",
            "Right Temple Block": "right_temple_block_correct", "Solar Plexus Thrust": "solar_plexus_thrust_correct"
        }
        self.selected_form = ctk.StringVar(value="Choose Arnis Form")
        self.form_button = ctk.CTkOptionMenu(
            self.controls_panel,
            variable=self.selected_form,
            values=list(self.practice_stances.keys()),
            command=self.on_action_selected,
            fg_color="#3498db",
            button_color="#2980b9",
            button_hover_color="#21618c",
            corner_radius=20,
            font=("Inter", 14)
        )
        self.form_button.pack(fill="x", pady=5, padx=10)
        
        # Separator
        ctk.CTkFrame(self.controls_panel, height=2, fg_color="#bdc3c7").pack(fill="x", pady=15, padx=15)
        
        self.status_label = ctk.CTkLabel(self.controls_panel, text="Status: Select a form", font=("Inter", 14), wraplength=220, text_color="#2c3e50")
        self.status_label.pack(fill="x", pady=5, anchor="w", padx=15)
        
        self.keras_status_label = ctk.CTkLabel(self.controls_panel, text="Keras: N/A (0.00)", font=("Inter", 12), text_color="#f39c12")
        self.keras_status_label.pack(fill="x", pady=5, anchor="w", padx=15)
        
        # System Status section
        ctk.CTkFrame(self.controls_panel, height=2, fg_color="#bdc3c7").pack(fill="x", pady=15, padx=15)
        
        system_frame = ctk.CTkFrame(self.controls_panel, corner_radius=10)
        system_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(system_frame, text="System Status", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        
        # FPS
        self.fps_label = ctk.CTkLabel(system_frame, text="FPS: --", font=("Inter", 18), text_color="#95a5a6")
        self.fps_label.pack(fill="x", pady=2, anchor="w", padx=10)
        
        # Camera status
        camera_status = "Connected" if self.cap.isOpened() else "Disconnected"
        camera_color = "#27ae60" if self.cap.isOpened() else "#e74c3c"
        self.camera_label = ctk.CTkLabel(system_frame, text=f"📷 Camera: {camera_status}", font=("Inter", 18), text_color=camera_color)
        self.camera_label.pack(fill="x", pady=2, anchor="w", padx=10)
        
        # Model status
        self.model_label = ctk.CTkLabel(system_frame, text="🤖 Model: Loaded", font=("Inter", 18), text_color="#27ae60")
        self.model_label.pack(fill="x", pady=2, anchor="w", padx=10, pady=(0, 10))
        
        # FPS tracking
        self.frame_times = []
        self.last_fps_update = time.time()
        
        self.view_all_results_button = ctk.CTkButton(
            self.controls_panel,
            text="View All Results",
            command=self.open_results_window,
            fg_color="#3498db",
            hover_color="#2980b9",
            corner_radius=20,
            font=("Inter", 14)
        )
        self.view_all_results_button.pack(fill="x", pady=10, side="bottom", padx=10)

        print("[DEBUG-INIT] System status added to controls panel...")

        print("[DEBUG-INIT] Starting video thread...")
        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()
        
        print("[DEBUG-INIT] Setting window geometry...")
        #set window size and center it (must be done together)
        width = int(self.screen_width * 0.8)
        height = int(self.screen_height * 0.8)
        x = (self.screen_width // 2) - (width // 2)
        y = (self.screen_height // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
        print("[DEBUG-INIT] Showing window...")
        
        # Destrfor professional transition feel
        import time
        time.sleep(1.0)  # Increased from 0.5 to 1.0 secondme.destroy()
        
        # Wait minimum time for professional feel
        import time
        time.sleep(0.5)
        
        self.toast.show(f"Welcome, {self.current_user['name']}!", "success", duration=2000)
        
        print("[DEBUG-INIT] Initialization complete, starting mainloop...")
        self.window.mainloop()
    
    @staticmethod
    def center_window(window, width=None, height=None):
        window.update_idletasks()
        if width is None or height is None:
            width = window.winfo_width()
            height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f"+{x}+{y}")

    def draw_text_with_bg(self, img, text, pos, font_face, font_scale, text_color, bg_color, thickness):
        (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        top_left = (pos[0], pos[1] - text_h - baseline)
        bottom_right = (pos[0] + text_w, pos[1] + baseline)
        cv2.rectangle(img, top_left, bottom_right, bg_color, cv2.FILLED)
        cv2.putText(img, text, (pos[0], pos[1]), font_face, font_scale, text_color, thickness)

    def resize_and_pad(self, img, size, pad_color=0):
        h, w, _ = img.shape; sw, sh = size
        if w == 0 or h == 0 or sw == 0 or sh == 0: return np.zeros((sh, sw, 3), dtype=np.uint8)
        aspect = w / h; canvas_aspect = sw / sh
        if aspect > canvas_aspect:
            new_w = sw; new_h = int(new_w / aspect)
            pad_top = (sh - new_h) // 2; pad_bot = sh - new_h - pad_top
            pad_left, pad_right = 0, 0
        else:
            new_h = sh; new_w = int(new_h * aspect)
            pad_left = (sw - new_w) // 2; pad_right = sw - new_w - pad_left
            pad_top, pad_bot = 0, 0
        interp = cv2.INTER_AREA if new_w < w or new_h < h else cv2.INTER_LINEAR
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[pad_color]*3)
        return padded_img
    
    def video_loop(self):
        COLOR_DEFAULT = (255, 0, 0); COLOR_CORRECT = (0, 255, 0); COLOR_ERROR = (0, 0, 255)
        COLOR_PROMPT = (0, 255, 255); COLOR_WHITE = (255, 255, 255); COLOR_BLACK = (0, 0, 0)
        COLOR_BG_TRANSPARENT = (0, 0, 0)

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            #balance: 480x360 provides better mediapipe accuracy without major performance hit
            #360x270 was too small and caused tracking issues
            processing_frame = cv2.resize(frame, (480, 360))
            
            #optimization: run ml inference and stick detection less frequently
            #but run mediapipe pose every frame for smooth skeleton
            run_full_ml = (self.frame_counter - self.last_ml_inference_frame) >= self.ml_inference_interval
            run_stick_detection = (self.frame_counter - self.last_stick_detection_frame) >= self.stick_detection_interval
            
            analysis_results = self.analyzer.process_frame(
                processing_frame, 
                skip_ml_inference=not run_full_ml,
                skip_stick_detection=not run_stick_detection
            )
            if analysis_results:
                self.last_known_results = analysis_results
                if run_full_ml:
                    self.last_ml_inference_frame = self.frame_counter
                if run_stick_detection:
                    self.last_stick_detection_frame = self.frame_counter

            feedback_x = processing_frame.shape[1] - 270; feedback_y = 30
            
            keras_status_text = "Keras: N/A (0.00)"
            if self.last_known_results:
                result = self.last_known_results[0]
                predicted_class = result['predicted_class']
                predicted_class = re.sub(r'^\d+\.\s*', '', predicted_class)
                confidence = result['confidence']
                pretty_class_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
                keras_status_text = f"Keras: {pretty_class_name} ({confidence:.2f})"
                if confidence > 0.60: self.keras_status_label.configure(text_color="#27ae60")
                elif confidence > 0.40: self.keras_status_label.configure(text_color="#f39c12")
                else: self.keras_status_label.configure(text_color="#e74c3c")
            self.keras_status_label.config(text=keras_status_text)
            
            if self.last_known_results:
                result = self.last_known_results[0]
                x1, y1, x2, y2 = result['bbox']
                person_id = result['id']
                
                draw_color = COLOR_ERROR; box_color = COLOR_DEFAULT; is_correct = False
                error_messages = []

                if self.target_form:
                    predicted_class = result['predicted_class']
                    predicted_class = re.sub(r'^\d+\.\s*', '', predicted_class)
                    confidence = result['confidence']
                    
                    if predicted_class.strip() == self.target_form.strip() and confidence > 0.60:
                        is_correct = True
                        draw_color = COLOR_CORRECT
                        box_color = COLOR_CORRECT
                        current_state = 'correct'
                    else:
                        is_correct = False
                        current_state = 'incorrect'
                    
                    #state transition tracking (fixed logic)
                    if self.current_session_id:
                        if current_state != self.last_pose_state:
                            #state changed - log the previous state if held long enough
                            if self.last_pose_state is not None and self.state_frame_count >= self.MIN_STATE_FRAMES:
                                is_correct = (self.last_pose_state == 'correct')
                                self.save_performance(result, is_correct=is_correct)
                                print(f"[ATTEMPT] {'Correct' if is_correct else 'Incorrect'} attempt completed ({self.state_frame_count} frames)")
                            
                            #start tracking new state
                            self.last_pose_state = current_state
                            self.state_frame_count = 1
                        else:
                            #same state - increment counter
                            self.state_frame_count += 1
                            
                            #timeout detection for stuck incorrect states
                            if self.state_frame_count >= self.MAX_STATE_DURATION and current_state == 'incorrect':
                                self.save_performance(result, is_correct=False)
                                print(f"[TIMEOUT] Logged failed attempt after {self.state_frame_count} frames (pose held too long)")
                                #reset state to allow fresh attempt
                                self.last_pose_state = None
                                self.state_frame_count = 0
                
                cv2.rectangle(processing_frame, (x1, y1), (x2, y2), box_color, 2)
                
                if result['stick_endpoints']:
                    pt1, pt2 = result['stick_endpoints']
                    cv2.line(processing_frame, pt1, pt2, COLOR_PROMPT, 4)

                user_display_name = self.current_user['name'] if self.current_user else f"Person {person_id}"
                self.draw_text_with_bg(img=processing_frame, text=user_display_name, pos=(x1, y1 - 10), font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, text_color=COLOR_BLACK, bg_color=COLOR_WHITE, thickness=2)

                if result.get('landmarks_absolute'):
                    landmarks_abs = result['landmarks_absolute']
                    person_id = result['id']
                    
                    frame_h, frame_w = processing_frame.shape[:2]
                    
                    #determine drawing color based on correctness
                    landmark_color = (0, 255, 0) if is_correct else (0, 0, 255)
                    connection_color = (0, 255, 0) if is_correct else (0, 0, 255)
                    
                    #draw landmarks with boundary checking
                    visible_landmarks = set()
                    for idx, (lx, ly, lz) in enumerate(landmarks_abs):
                        # Only draw if within frame bounds
                        if 0 <= lx < frame_w and 0 <= ly < frame_h:
                            cv2.circle(processing_frame, (int(lx), int(ly)), 4, landmark_color, -1, lineType=cv2.LINE_AA)
                            visible_landmarks.add(idx)
                    
                    #draw connections only if both endpoints are visible
                    pose_connections = self.analyzer.mp_pose.POSE_CONNECTIONS
                    for connection in pose_connections:
                        start_idx, end_idx = connection
                        
                        # Check if both landmarks are valid and visible
                        if (start_idx < len(landmarks_abs) and end_idx < len(landmarks_abs) and
                            start_idx in visible_landmarks and end_idx in visible_landmarks):
                            
                            start_pt = (int(landmarks_abs[start_idx][0]), int(landmarks_abs[start_idx][1]))
                            end_pt = (int(landmarks_abs[end_idx][0]), int(landmarks_abs[end_idx][1]))
                            cv2.line(processing_frame, start_pt, end_pt, connection_color, 3, lineType=cv2.LINE_AA)

                if self.target_form:
                    overlay = processing_frame.copy()
                    cv2.rectangle(overlay, (feedback_x - 10, feedback_y - 20), (processing_frame.shape[1] - 10, feedback_y + 150), COLOR_BG_TRANSPARENT, -1)
                    alpha = 0.6
                    processing_frame = cv2.addWeighted(overlay, alpha, processing_frame, 1 - alpha, 0)
                    
                    if is_correct:
                        cv2.putText(processing_frame, "Correct!", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CORRECT, 2)
                    else:
                        error_display_list = []
                        if result['grip_angle'] is not None:
                            target_min, target_max = 80, 120 
                            if not (target_min <= result['grip_angle'] <= target_max):
                                feedback = "Extend stick" if result['grip_angle'] < target_min else "Retract stick"
                                error_display_list.append(f"Grip: {feedback}")
                        
                        error_display_list.extend(error_messages)

                        if error_display_list:
                            cv2.putText(processing_frame, "Feedback:", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PROMPT, 2)
                            for i, msg in enumerate(error_display_list[:4]):
                                cv2.putText(processing_frame, msg, (feedback_x, feedback_y + 30 + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ERROR, 2)
                        else:
                            pretty_form_name = self.form_button.cget('text')
                            if pretty_form_name != "Choose Arnis Form":
                                cv2.putText(processing_frame, f"Adjust to Form:", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PROMPT, 2)
                                cv2.putText(processing_frame, pretty_form_name, (feedback_x, feedback_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 2)
            
            canvas_width = self.video_canvas.winfo_width(); canvas_height = self.video_canvas.winfo_height()
            final_frame = self.resize_and_pad(processing_frame, size=(canvas_width, canvas_height))
            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(final_frame)
            
            self.frame_counter += 1
            
            #update fps
            self.update_fps_display(time.time())
            
            time.sleep(0.01)

    def process_queue(self):
        try:
            frame = self.queue.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(self.video_canvas.winfo_width()/2, self.video_canvas.winfo_height()/2, image=imgtk, anchor="center")
            self.tk_image = imgtk
        except queue.Empty: pass
        finally: self.window.after(30, self.process_queue)

    def on_canvas_resize(self, event):
        self.process_queue() 

    def on_action_selected(self, pretty_name):
        self.target_form = self.practice_stances[pretty_name]
        self.selected_form.set(pretty_name)
        self.status_label.config(text=f"Status: Analyzing '{pretty_name}'")
        self.status_bar.set_status(f"Practicing: {pretty_name}")
        print(f"[INFO] targeting: '{self.target_form}'")
        
        self.toast.show(f"Now practicing: {pretty_name}", "info", duration=2000)

        if self.current_user and not self.current_session_id:
            self.start_session()
    
    def show_user_selection(self):
        print("[DEBUG-MAIN] Calling show_user_dialog...")
        selected = show_user_dialog(self.window, self.db)
        print(f"[DEBUG-MAIN] Dialog returned: {selected}")
        print(f"[DEBUG-MAIN] Selected type: {type(selected)}")
        
        if selected:
            print(f"[DEBUG-MAIN] User selected, setting current_user...")
            self.current_user = selected
            print(f"[DEBUG-MAIN] current_user set to: {self.current_user}")
            print(f"[DEBUG-MAIN] User name: {self.current_user['name']}")
            print(f"[DEBUG-MAIN] User ID: {self.current_user['id']}")
            print(f"[DEBUG-MAIN] User active: {self.current_user.get('is_active', 'KEY NOT FOUND')}")
        else:
            print("[DEBUG-MAIN] No user selected (selected is None/False)")
            self.current_user = None
    
    def start_session(self):
        if not self.current_user:
            return
        
        self.current_session_id = self.db.start_session(
            user_id=self.current_user['id'],
            target_pose=self.target_form
        )
        print(f"[INFO] session {self.current_session_id} started")

        self.session_status_label.configure(text=f"Session #{self.current_session_id} - active", text_color="#27ae60")
        self.start_session_btn.configure(state="disabled")
        self.end_session_btn.configure(state="normal")
        
        #show toast notification
        self.toast.show(f"Session #{self.current_session_id} started", "success", duration=2000)
        self.status_bar.set_status("Session active - Good luck!")
    
    def manual_start_session(self):
        if not self.target_form:
            self.toast.show("Please select a target form first", "warning", duration=3000)
            return
        self.start_session()
    
    def end_session(self):
        if self.current_session_id:
            self.db.end_session(self.current_session_id)
            print(f"[INFO] session {self.current_session_id} ended")

            summary = self.db.get_session_summary(self.current_session_id)
            
            #show summary as toast
            if summary['total_attempts'] > 0:
                accuracy = (summary['correct_attempts']/summary['total_attempts']*100)
                msg = f"Session Complete! {summary['correct_attempts']}/{summary['total_attempts']} correct ({accuracy:.1f}%)"
                toast_type = "success" if accuracy >= 70 else "warning" if accuracy >= 50 else "info"
            else:
                msg = "Session ended - No attempts recorded"
                toast_type = "info"
            
            self.toast.show(msg, toast_type, duration=5000)
            self.status_bar.set_status("Session ended")

            self.current_session_id = None

            self.session_status_label.configure(text="No active session", text_color="#f39c12")
            self.start_session_btn.configure(state="normal")
            self.end_session_btn.configure(state="disabled")
    
    def save_performance(self, result, is_correct):
        if not self.current_session_id:
            return
        
        predicted_class = result.get('predicted_class', 'N/A')
        predicted_class = re.sub(r'^\d+\.\s*', '', predicted_class)
        confidence = result.get('confidence', 0.0)
        joint_angles = result.get('live_angles')
        grip_angle = result.get('grip_angle')
        stick_detected = result.get('stick_endpoints') is not None

        self.db.save_performance(
            session_id=self.current_session_id,
            user_id=self.current_user['id'],
            pose_detected=predicted_class,
            confidence=float(confidence),
            is_correct=is_correct,
            joint_angles=joint_angles,
            grip_angle=float(grip_angle) if grip_angle else None,
            stick_detected=stick_detected
        )
    
    def open_results_window(self):
        results_window = ResultsWindow(self.window, db_manager=self.db, current_user=self.current_user)
        self.center_window(results_window, 1200, 700)
    
    def on_closing(self):
        print("[INFO] closing...")
        self.is_running = False
        time.sleep(0.5)

        self.end_session()

        self.db.close()
        
        self.analyzer.close()
        self.cap.release()
        self.window.destroy()
    
    def on_user_selected(self, username):
        pass
    
    def reset_feedback(self):
        self.target_form = None
        self.selected_form.set("Choose Arnis Form")
        self.status_label.config(text="Status: Select a form")
    
    def update_fps_display(self, frame_time=None):
        """Update FPS display in controls panel"""
        if frame_time is None:
            frame_time = time.time()
        
        self.frame_times.append(frame_time)
        
        # Keep last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Update every 0.5 seconds
        if time.time() - self.last_fps_update > 0.5:
            if len(self.frame_times) > 1:
                fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                
                # Color code FPS
                if fps >= 25:
                    color = "#27ae60"
                elif fps >= 15:
                    color = "#f39c12"
                else:
                    color = "#e74c3c"
                
                self.fps_label.configure(
                    text=f"FPS: {fps:.0f}",
                    text_color=color
                )
            
            self.last_fps_update = time.time()

if __name__ == "__main__":
    #windows: set app id so taskbar icon shows properly
    try:
        from ctypes import windll
        #set unique app id for windows taskbar
        windll.shell32.SetCurrentProcessExplicitAppUserModelID('TuroArnis.ArnisFormCorrection.1.0')
    except:
        pass  #not on windows or failed
    
    root = ctk.CTk()  # CustomTkinter window
    
    #set icon before creating the gui
    try:
        from app.utils.resource_path import get_resource_path
        icon_path = get_resource_path('app/assets/TA.ico')
        print(f"[DEBUG-ICON-MAIN] Icon path resolved to: {icon_path}")
        print(f"[DEBUG-ICON-MAIN] Icon exists: {os.path.exists(icon_path)}")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
            print(f"[DEBUG-ICON-MAIN] Icon set successfully")
        else:
            print(f"[DEBUG-ICON-MAIN] Icon file not found at {icon_path}")
    except Exception as e:
        print(f"[WARNING] Could not set icon: {e}")
    
    app = TuroArnisGUI(root, "TuroArnis - Arnis Form Correction")