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

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

if getattr(sys, 'frozen', False):
    # Frozen state (PyInstaller)
    # Add the internal directory to sys.path to allow 'from app.gui' imports
    base_path = sys._MEIPASS
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    # Also add 'app' directory to path to allow imports like 'from utils...' if code relies on 'app' being root
    app_path = os.path.join(base_path, 'app')
    if os.path.exists(app_path) and app_path not in sys.path:
        sys.path.insert(0, app_path)
    
    # Also valid for onedir mode where _internal might be the root for packages
    # Ensure current directory semantics
else:
    # Dev state
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # Add project root (TuroArnis/) to sys.path so we can import 'app'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.gui.results_window import ResultsWindow
from app.gui.user_dialog import show_user_dialog
from app.gui.toast import ToastNotification
from app.gui.loading_spinner import LoadingSpinner
from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.database.db_manager import DatabaseManager
from app.utils.resource_path import get_resource_path, get_app_data_path

DEBUG_MODE = False  

class TuroArnisGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(fg_color="#2c3e50")
        
        #set app icon
        icon_path = get_resource_path('app/assets/TA.ico')
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
        
        self.window.update_idletasks()
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        
        #splash screen
        self.splash_frame = ctk.CTkFrame(self.window, fg_color="#74b9ff", corner_radius=0)
        self.splash_frame.pack(fill="both", expand=True)
        
        splash_content = ctk.CTkFrame(self.splash_frame, fg_color="#74b9ff")
        splash_content.place(relx=0.5, rely=0.5, anchor="center")
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
        except:
            pass
        
        ctk.CTkLabel(
            splash_content,
            text="TuroArnis",
            font=("Inter", 48, "bold"),
            fg_color="#74b9ff",
            text_color="black"
        ).pack(pady=(0, 5))
        
        ctk.CTkLabel(
            splash_content,
            text="Arnis Form Correction",
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
        
        #window size (80% of screen)
        width = int(self.screen_width * 0.8)
        height = int(self.screen_height * 0.8)
        x = (self.screen_width - width) // 2
        y = (self.screen_height - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.update()

        #database
        db_path = os.path.join(get_app_data_path(), 'turoarnis.db')
        self.db = DatabaseManager(db_path)
        self.current_user = None
        self.current_session_id = None
        
        if not DEBUG_MODE:
            for i in range(30):
                time.sleep(0.1)
                self.window.update()
            
            self.splash_status.configure(text="Loading user management...")
            self.window.update()
            for i in range(10):
                time.sleep(0.1)
                self.window.update()
            self.splash_frame.pack_forget()
            self.window.update()
        else:
            #debug mode: skip splash instantly
            self.splash_frame.pack_forget()
            self.window.update()
        
        self.show_user_selection()
        
        if not self.current_user:
            try:
                self.window.destroy()
            except:
                pass
            return
        
        if not DEBUG_MODE:
            try:
                if self.splash_frame.winfo_exists():
                    self.splash_frame.pack(fill="both", expand=True)
                    self.window.update()
            except:
                pass

        #frame processing config
        self.frame_counter = 0
        self.processing_interval = 1
        self.ml_inference_interval = 8
        self.stick_detection_interval = 4
        self.last_known_results = []
        self.last_ml_inference_frame = 0
        self.last_stick_detection_frame = 0
        
        #state tracking
        self.MIN_STATE_FRAMES = 10
        self.MAX_STATE_DURATION = 300
        self.last_pose_state = None
        self.state_frame_count = 0
        
        #landmark smoothing buffer (reduces jitter)
        self.landmark_smooth_buffer = {}  #dictionary keyed by person_id
        self.smooth_window = 3  #smooth over 3 frames
        self.last_person_bbox = {}  #track bounding boxes to detect movement

        #initialize ux components
        print("[DEBUG-INIT] Initializing UX components...")
        self.toast = ToastNotification(self.window)

        #load models
        self.splash_status.configure(text="Loading stick detector...")
        self.window.update()
        time.sleep(0.3)
        
        stick_model_relative = 'app/models/weights/best.pt'
        stick_model_path = get_resource_path(stick_model_relative)
        
        self.splash_status.configure(text="Loading AI models...")
        self.window.update()
        time.sleep(0.3)
        
        self.analyzer = PoseAnalyzer(
            detection_interval=self.processing_interval,
            stick_model_path=stick_model_path if os.path.exists(stick_model_path) else None,
            debug_stick=False
        )
        
        #initialize feedback analyzer
        self.feedback_analyzer = FeedbackAnalyzer()
        
        time.sleep(0.3)
        
        #camera init
        self.splash_status.configure(text="Connecting to camera...")
        self.window.update()
        
        self.cap = cv2.VideoCapture(0)
        
        #safely destroy splash frame
        try:
            self.splash_progress.stop()
            if self.splash_frame.winfo_exists():
                self.splash_frame.destroy()
        except:
            pass
        
        if not self.cap.isOpened():
            self.toast.show("Camera not detected.", "error", duration=5000)
        
        #gui setup
        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        
        main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        main_container.pack(fill="both", expand=True)
        
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=0)
        main_container.grid_columnconfigure(1, weight=1) 

        #create frame to hold video canvas
        video_frame = ctk.CTkFrame(main_container, fg_color="black")
        video_frame.grid(row=0, column=1, sticky="nsew")
        
        self.video_canvas = tk.Canvas(video_frame, background='black', highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True)
        self.video_canvas.bind('<Configure>', self.on_canvas_resize)
        self.tk_image = None

        self.controls_panel = ctk.CTkFrame(main_container, width=250, corner_radius=0, fg_color="white")
        self.controls_panel.grid(row=0, column=0, sticky="nsew")
        self.controls_panel.grid_propagate(False) 
        
        ctk.CTkLabel(self.controls_panel, text="Controls", font=("Inter", 18, "bold"), text_color="#2c3e50").pack(pady=(10, 10), anchor="w", padx=15)

        #session frame (includes user info and form selection)
        session_frame = ctk.CTkFrame(self.controls_panel, corner_radius=10, fg_color="white")
        session_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(session_frame, text="Session", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        
        ctk.CTkLabel(session_frame, text=self.current_user['name'], font=("Inter", 14, "bold"), text_color="#27ae60").pack(anchor="w", padx=10, pady=(0, 5))
        
        self.session_status_label = ctk.CTkLabel(session_frame, text="No active session", font=("Inter", 12), text_color="#f39c12")
        self.session_status_label.pack(anchor="w", pady=2, padx=10)
        
        #session timer
        self.session_start_time = None
        self.timer_update_id = None
        self.timer_label = ctk.CTkLabel(
            session_frame,
            text="⏱️ 00:00",
            font=("Inter", 16, "bold"),
            text_color="#3498db"
        )
        self.timer_label.pack(anchor="w", pady=(5, 10), padx=10)
        
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
            session_frame,
            variable=self.selected_form,
            values=list(self.practice_stances.keys()),
            command=self.on_action_selected,
            fg_color="#3498db",
            button_color="#2980b9",
            button_hover_color="#21618c",
            corner_radius=10,
            font=("Inter", 14)
        )
        self.form_button.pack(fill="x", pady=5, padx=10)

        # Viewpoint Selection (New for GCN)
        ctk.CTkLabel(session_frame, text="Camera Viewpoint", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        self.selected_viewpoint = ctk.StringVar(value="front")
        self.viewpoint_menu = ctk.CTkOptionMenu(
            session_frame,
            variable=self.selected_viewpoint,
            values=["front", "left", "right"],
            command=self.on_viewpoint_selected,
            fg_color="#34495e",
            button_color="#2c3e50",
            button_hover_color="#1a252f",
            corner_radius=10,
            font=("Inter", 14)
        )
        self.viewpoint_menu.pack(fill="x", pady=5, padx=10)
        
        session_btn_frame = ctk.CTkFrame(session_frame, fg_color="transparent")
        session_btn_frame.pack(fill="x", pady=5, padx=10)
        
        self.start_session_btn = ctk.CTkButton(session_btn_frame, text="Start", command=self.manual_start_session, fg_color="#27ae60", hover_color="#229954", width=100, corner_radius=10, font=("inter", 14), text_color="white")
        self.start_session_btn.pack(side="left", padx=2)
        
        self.end_session_btn = ctk.CTkButton(session_btn_frame, text="End", command=self.end_session, fg_color="#e74c3c", hover_color="#c0392b", width=100, corner_radius=10, font=("inter", 14), state="disabled", text_color="white")
        self.end_session_btn.pack(side="left", padx=2)
        
        ctk.CTkFrame(self.controls_panel, height=2, fg_color="#bdc3c7").pack(fill="x", pady=10, padx=15)
        
        #system status frame (confidence, camera, model only)
        system_frame = ctk.CTkFrame(self.controls_panel, corner_radius=10, fg_color="white")
        system_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(system_frame, text="System Status", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        
        #confidence progress bar
        confidence_frame = ctk.CTkFrame(system_frame, fg_color="transparent")
        confidence_frame.pack(fill="x", pady=(5, 5), padx=10)
        
        ctk.CTkLabel(
            confidence_frame,
            text="Confidence:",
            font=("Inter", 11),
            text_color="#7f8c8d"
        ).pack(side="left", padx=(0, 5))
        
        self.confidence_progress = ctk.CTkProgressBar(
            confidence_frame,
            mode='determinate',
            progress_color="#27ae60",
            height=10,
            width=120
        )
        self.confidence_progress.pack(side="left", fill="x", expand=True)
        self.confidence_progress.set(0)
        
        self.confidence_percent_label = ctk.CTkLabel(
            confidence_frame,
            text="0%",
            font=("Inter", 11, "bold"),
            text_color="#27ae60"
        )
        self.confidence_percent_label.pack(side="left", padx=(5, 0))
        
        camera_status = "Connected" if self.cap.isOpened() else "Disconnected"
        camera_color = "#27ae60" if self.cap.isOpened() else "#e74c3c"
        self.camera_label = ctk.CTkLabel(system_frame, text=f"Camera: {camera_status}", font=("Inter", 11), text_color=camera_color, anchor="w")
        self.camera_label.pack(fill="x", pady=2, padx=10)
        
        self.model_label = ctk.CTkLabel(system_frame, text="Model: Loaded", font=("Inter", 11), text_color="#27ae60", anchor="w")
        self.model_label.pack(fill="x", padx=10, pady=(0, 10))
        
        self.frame_times = []
        self.last_fps_update = time.time()
        
        #activity log and button - pack button first (bottom), then activity label above it
        self.view_all_results_button = ctk.CTkButton(
            self.controls_panel,
            text="View All Results",
            command=self.open_results_window,
            fg_color="#3498db",
            hover_color="#2980b9",
            corner_radius=10,
            font=("Inter", 14),
            text_color="white"
        )
        self.view_all_results_button.pack(fill="x", pady=(0, 10), side="bottom", padx=10)
        
        self.activity_label = ctk.CTkLabel(
            self.controls_panel, 
            text="Ready", 
            font=("Inter", 12), 
            text_color="#7f8c8d",
            wraplength=220
        )
        self.activity_label.pack(fill="x", pady=(10, 5), padx=10, side="bottom")

        #start video thread
        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        #keyboard shortcuts
        self.window.bind("<space>", self.toggle_session_keybind)
        self.window.bind("<Control-q>", lambda e: self.on_closing())
        self.window.bind("<Control-r>", lambda e: self.open_results_window())
        self.window.bind("<F11>", self.toggle_fullscreen)
        self.is_fullscreen = False
        
        self.process_queue()
        
        width = int(self.screen_width * 0.8)
        height = int(self.screen_height * 0.8)
        x = (self.screen_width // 2) - (width // 2)
        y = (self.screen_height // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
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
            processing_frame = cv2.resize(frame, (480, 360))
            
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
            
            prediction_text = "Prediction: N/A (0.00)"
            if self.last_known_results:
                result = self.last_known_results[0]
                predicted_class = result['predicted_class']
                predicted_class = re.sub(r'^\d+\.\s*', '', predicted_class)
                confidence = result['confidence']
                pretty_class_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
                prediction_text = f"Prediction: {pretty_class_name} ({confidence:.2f})"
                
            #update confidence progress bar
                self.confidence_progress.set(confidence)
                self.confidence_percent_label.configure(text=f"{int(confidence * 100)}%")
                
                #color code based on confidence
                if confidence > 0.50:
                    self.confidence_progress.configure(progress_color="#27ae60")
                    self.confidence_percent_label.configure(text_color="#27ae60")
                elif confidence > 0.40:
                    self.confidence_progress.configure(progress_color="#f39c12")
                    self.confidence_percent_label.configure(text_color="#f39c12")
                else:
                    self.confidence_progress.configure(progress_color="#e74c3c")
                    self.confidence_percent_label.configure(text_color="#e74c3c")
            else:
                #reset when no results
                self.confidence_progress.set(0)
                self.confidence_percent_label.configure(text="0%", text_color="#95a5a6")
                
            
            if self.last_known_results:
                result = self.last_known_results[0]
                x1, y1, x2, y2 = result['bbox']
                person_id = result['id']
                
                draw_color = COLOR_ERROR; box_color = COLOR_DEFAULT; is_correct = False

                if self.target_form:
                    #use feedback analyzer to determine correctness
                    feedback = self.feedback_analyzer.analyze(result, self.target_form, viewpoint=self.selected_viewpoint.get())
                    is_correct = feedback['is_correct']
                    
                    if is_correct:
                        draw_color = COLOR_CORRECT
                        box_color = COLOR_CORRECT
                        current_state = 'correct'
                    else:
                        current_state = 'incorrect'

                    # Draw visual correction arrows
                    if result.get('landmarks_absolute'):
                         self.analyzer.draw_visual_cues(processing_frame, feedback, result['landmarks_absolute'])

                    #state tracking
                    if self.current_session_id:
                        if current_state != self.last_pose_state:
                            if self.last_pose_state is not None and self.state_frame_count >= self.MIN_STATE_FRAMES:
                                is_correct = (self.last_pose_state == 'correct')
                                self.save_performance(result, is_correct=is_correct)
                            
                            self.last_pose_state = current_state
                            self.state_frame_count = 1
                        else:
                            self.state_frame_count += 1
                            
                            #timeout for stuck states
                            if self.state_frame_count >= self.MAX_STATE_DURATION and current_state == 'incorrect':
                                self.save_performance(result, is_correct=False)
                                self.last_pose_state = None
                                self.state_frame_count = 0
                
                cv2.rectangle(processing_frame, (x1, y1), (x2, y2), box_color, 2)
                
                if result['stick_endpoints']:
                    pt1, pt2 = result['stick_endpoints']
                    cv2.line(processing_frame, pt1, pt2, COLOR_PROMPT, 4)

                if result.get('landmarks_absolute'):
                    landmarks_abs = result['landmarks_absolute']
                    person_id = result['id']
                    
                    frame_h, frame_w = processing_frame.shape[:2]
                    
                    landmark_color = (0, 255, 0) if is_correct else (0, 0, 255)
                    connection_color = (0, 255, 0) if is_correct else (0, 0, 255)
                    
                    visible_landmarks = set()
                    for idx, (lx, ly, lz) in enumerate(landmarks_abs):
                        if 0 <= lx < frame_w and 0 <= ly < frame_h:
                            cv2.circle(processing_frame, (int(lx), int(ly)), 4, landmark_color, -1, lineType=cv2.LINE_AA)
                            visible_landmarks.add(idx)
                    
                    pose_connections = self.analyzer.mp_pose.POSE_CONNECTIONS
                    for connection in pose_connections:
                        start_idx, end_idx = connection
                        
                        if (start_idx < len(landmarks_abs) and end_idx < len(landmarks_abs) and
                            start_idx in visible_landmarks and end_idx in visible_landmarks):
                            
                            start_pt = (int(landmarks_abs[start_idx][0]), int(landmarks_abs[start_idx][1]))
                            end_pt = (int(landmarks_abs[end_idx][0]), int(landmarks_abs[end_idx][1]))
                            cv2.line(processing_frame, start_pt, end_pt, connection_color, 3, lineType=cv2.LINE_AA)

                if self.target_form:
                    #use feedback analyzer to get detailed feedback
                    if 'feedback' not in locals(): # In case we skipped the block above (unlikely but safe)
                        feedback = self.feedback_analyzer.analyze(result, self.target_form, viewpoint=self.selected_viewpoint.get())

                    prioritized_messages = self.feedback_analyzer.get_prioritized_messages(feedback, max_messages=3)
                    
                    #opencv feedback rendering - compact size
                    box_width = 220
                    box_height = 110
                    box_x = processing_frame.shape[1] - box_width - 10
                    box_y = 10
                    
                    #semi-transparent background
                    overlay = processing_frame.copy()
                    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (30, 30, 30), -1)
                    processing_frame = cv2.addWeighted(overlay, 0.8, processing_frame, 0.2, 0)
                    
                    #border color based on state
                    if feedback['is_correct']:
                        border_color = (0, 200, 0)
                    elif feedback.get('severity') == 'critical':
                        border_color = (0, 0, 200)
                    else:
                        border_color = (200, 150, 50)
                    cv2.rectangle(processing_frame, (box_x, box_y), (box_x + box_width, box_y + box_height), border_color, 2)
                    
                    content_x = box_x + 10
                    content_y = box_y + 20
                    
                    if feedback['is_correct']:
                        cv2.putText(processing_frame, "Perfect Form!", (content_x, content_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(processing_frame, "Maintain position", (content_x, content_y + 22), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        if prioritized_messages:
                            cv2.putText(processing_frame, "Feedback:", (content_x, content_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            msg_y = content_y + 24
                            for i, (message, msg_type) in enumerate(prioritized_messages):
                                if msg_y > box_y + box_height - 10:
                                    break
                                display_msg = message[:28] + ".." if len(message) > 28 else message
                                cv2.putText(processing_frame, display_msg, (content_x, msg_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                                msg_y += 22
            
            
            canvas_width = self.video_canvas.winfo_width(); canvas_height = self.video_canvas.winfo_height()
            final_frame = self.resize_and_pad(processing_frame, size=(canvas_width, canvas_height))
            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(final_frame)
            
            self.frame_counter += 1
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
        self.activity_label.configure(text=f"▶ Practicing: {pretty_name}", text_color="#3498db")

        if self.current_user and not self.current_session_id:
            self.start_session()

    def on_viewpoint_selected(self, viewpoint):
        """Callback when user changes camera viewpoint"""
        if hasattr(self, 'analyzer'):
            self.analyzer.set_viewpoint(viewpoint)
            self.activity_label.configure(text=f"👁 Viewpoint switched to: {viewpoint.title()}", text_color="#3498db")

        
        self.activity_label.configure(text=f"▶ Practicing: {pretty_name}", text_color="#3498db")

        if self.current_user and not self.current_session_id:
            self.start_session()
    
    
    def show_user_selection(self):
        if DEBUG_MODE:
            #auto-select last active user or create test user
            cursor = self.db.conn.cursor()
            cursor.execute('SELECT * FROM users WHERE is_active = 1 ORDER BY id DESC LIMIT 1')
            user = cursor.fetchone()
            
            if user:
                self.current_user = {
                    'id': user[0],
                    'name': user[1],
                    'created_at': user[2],
                    'is_active': user[3]
                }
                print(f"[DEBUG] Auto-selected user: {self.current_user['name']}")
            else:
                #create test user if none exists
                test_name = "TestUser"
                user_id = self.db.create_user(test_name)
                if user_id:
                    self.current_user = self.db.get_user_by_id(user_id)
                    print(f"[DEBUG] Created test user: {test_name}")
                else:
                    #test user already exists, fetch it
                    cursor.execute('SELECT * FROM users WHERE name = ?', (test_name,))
                    user = cursor.fetchone()
                    if user:
                        self.current_user = {
                            'id': user[0],
                            'name': user[1],
                            'created_at': user[2],
                            'is_active': user[3]
                        }
                        print(f"[DEBUG] Using existing TestUser")
        else:
            #normal mode: show user dialog
            selected = show_user_dialog(self.window, self.db)
            self.current_user = selected if selected else None
    
    def start_session(self):
        if not self.current_user:
            return
        
        self.current_session_id = self.db.start_session(
            user_id=self.current_user['id'],
            target_pose=self.target_form
        )

        self.session_status_label.configure(text=f"Session #{self.current_session_id} - active", text_color="#27ae60")
        self.start_session_btn.configure(state="disabled")
        self.end_session_btn.configure(state="normal")
        
        #start timer
        self.session_start_time = time.time()
        self.update_timer()
        
        self.activity_label.configure(text=f"● Session #{self.current_session_id} started", text_color="#27ae60")
    
    def manual_start_session(self):
        if not self.target_form:
            self.activity_label.configure(text="⚠ Select a target form first", text_color="#f39c12")
            return
        self.start_session()
    
    def end_session(self):
        if self.current_session_id:
            self.db.end_session(self.current_session_id)
            summary = self.db.get_session_summary(self.current_session_id)
            
            if summary['total_attempts'] > 0:
                accuracy = (summary['correct_attempts']/summary['total_attempts']*100)
                msg = f"Session Complete! {summary['correct_attempts']}/{summary['total_attempts']} correct ({accuracy:.1f}%)"
                toast_type = "success" if accuracy >= 70 else "warning" if accuracy >= 50 else "info"
            else:
                msg = "Session ended - No attempts recorded"
                toast_type = "info"
            
            color = "#27ae60" if toast_type == "success" else "#f39c12" if toast_type == "warning" else "#7f8c8d"
            self.activity_label.configure(text=msg, text_color=color)

            self.current_session_id = None
            
            #stop timer
            if self.timer_update_id:
                self.window.after_cancel(self.timer_update_id)
                self.timer_update_id = None
            self.session_start_time = None
            self.timer_label.configure(text="⏱️ 00:00")

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
        self.activity_label.configure(text="Ready", text_color="#7f8c8d")
    
    def update_fps_display(self, frame_time=None):
        if frame_time is None:
            frame_time = time.time()
        
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if time.time() - self.last_fps_update > 0.5:
            if len(self.frame_times) > 1:
                fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                
                if fps >= 25:
                    color = "#27ae60"
                elif fps >= 15:
                    color = "#f39c12"
                else:
                    color = "#e74c3c"
                

            
            self.last_fps_update = time.time()
    
    def update_timer(self):
        """Update session timer every second"""
        if self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.timer_label.configure(text=f"⏱️ {minutes:02d}:{seconds:02d}")
            
            # Schedule next update
            self.timer_update_id = self.window.after(1000, self.update_timer)
    
    def toggle_session_keybind(self, event):
        """Toggle session start/stop with spacebar"""
        if self.current_session_id:
            self.end_session()
        else:
            if self.target_form:
                self.start_session()
            else:
                self.activity_label.configure(text="⚠ Select a form first", text_color="#f39c12")
    
    def toggle_fullscreen(self, event):
        """Toggle fullscreen mode with F11"""
        self.is_fullscreen = not self.is_fullscreen
        self.window.attributes("-fullscreen", self.is_fullscreen)
        
        if self.is_fullscreen:
            self.activity_label.configure(text="Fullscreen (F11 to exit)", text_color="#7f8c8d")

if __name__ == "__main__":
    # Windows taskbar icon
    try:
        from ctypes import windll
        windll.shell32.SetCurrentProcessExplicitAppUserModelID('TuroArnis.ArnisFormCorrection.1.0')
    except:
        pass
    
    root = ctk.CTk()
    
    try:
        from app.utils.resource_path import get_resource_path
        icon_path = get_resource_path('app/assets/TA.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass
    
    app = TuroArnisGUI(root, "TuroArnis - Arnis Form Correction")
