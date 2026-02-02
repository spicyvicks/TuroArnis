import sys
import os
import cv2
import threading
import time
import re
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import queue
import numpy as np

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

#add project root to sys.path for dev execution
if not getattr(sys, 'frozen', False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.gui.results_window import ResultsWindow
from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.feedback_analyzer import FeedbackAnalyzer
from app.utils.resource_path import get_resource_path

class TuroArnisVideoGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(fg_color="#2c3e50")
        
        #set app icon for taskbar
        icon_path = get_resource_path('app/assets/TA.ico')
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
        
        self.window.update_idletasks()
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        
        #test user, no db
        self.current_user = {'id': 0, 'name': 'Test User (Video Mode)'}
        self.current_session_id = None

        self.frame_counter = 0
        self.processing_interval = 1
        self.last_known_results = []
        
        #performance optimization - skip heavy processing on some frames
        self.ml_inference_interval = 8  #run ML every 8 frames
        self.stick_detection_interval = 4  #run stick detection every 4 frames
        self.last_ml_inference_frame = 0
        self.last_stick_detection_frame = 0
        self.analysis_mode = "balanced"  #fast, balanced, detailed
        
        #state tracking
        self.last_pose_state = None
        self.state_frame_count = 0
        self.min_state_frames = 15

        #video playback state
        self.video_path = None
        self.cap = None
        self.video_fps = 30
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.is_video_loaded = False
        self.display_mode = "fit"  #fit (letterbox) or fill (crop)

        #use resource path for stick detector
        stick_model_relative = 'runs/pose/arnis_stick_detector/weights/best.pt'
        stick_model_path = get_resource_path(stick_model_relative)
        self.analyzer = PoseAnalyzer(
            detection_interval=self.processing_interval,
            stick_model_path=stick_model_path if os.path.exists(stick_model_path) else None,
            debug_stick=True
        )
        
        #initialize feedback analyzer
        self.feedback_analyzer = FeedbackAnalyzer()
        
        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        
        #main layout
        main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        main_container.pack(fill="both", expand=True)
        
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=0)
        main_container.grid_columnconfigure(1, weight=1)

        #video canvas
        video_frame = ctk.CTkFrame(main_container, fg_color="black")
        video_frame.grid(row=0, column=1, sticky="nsew")
        
        self.video_canvas = tk.Canvas(video_frame, background='black', highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True)
        self.video_canvas.bind('<Configure>', self.on_canvas_resize)
        self.tk_image = None

        #controls panel
        self.controls_panel = ctk.CTkFrame(main_container, width=250, corner_radius=0, fg_color="white")
        self.controls_panel.grid(row=0, column=0, sticky="nsew")
        self.controls_panel.grid_propagate(False)
        
        ctk.CTkLabel(self.controls_panel, text="Controls", font=("Inter", 18, "bold"), text_color="#2c3e50").pack(pady=(10, 10), anchor="w", padx=15)

        #user frame
        user_frame = ctk.CTkFrame(self.controls_panel, corner_radius=10, fg_color="white")
        user_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(user_frame, text="Current User", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        ctk.CTkLabel(user_frame, text=self.current_user['name'], font=("Inter", 14, "bold"), text_color="#3498db").pack(anchor="w", padx=10)
        ctk.CTkLabel(user_frame, text="Video Testing Mode", font=("Inter", 11), text_color="#7f8c8d").pack(anchor="w", padx=10, pady=(0, 10))

        #video file section
        video_frame_ctrl = ctk.CTkFrame(self.controls_panel, corner_radius=10, fg_color="white")
        video_frame_ctrl.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(video_frame_ctrl, text="Video File", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        
        self.video_name_label = ctk.CTkLabel(video_frame_ctrl, text="No video loaded", font=("Inter", 11), text_color="#7f8c8d", wraplength=200)
        self.video_name_label.pack(anchor="w", pady=(0, 5), padx=10)
        
        ctk.CTkButton(video_frame_ctrl, text="📂 Open Video", command=self.open_video_dialog, fg_color="#3498db", hover_color="#2980b9", corner_radius=10, font=("Inter", 14)).pack(fill="x", pady=(0, 10), padx=10)

        #playback controls section
        playback_frame = ctk.CTkFrame(self.controls_panel, corner_radius=10, fg_color="white")
        playback_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(playback_frame, text="Playback", font=("Inter", 12, "bold"), text_color="#7f8c8d").pack(anchor="w", pady=(5, 2), padx=10)
        
        btn_row = ctk.CTkFrame(playback_frame, fg_color="transparent")
        btn_row.pack(fill="x", pady=5, padx=10)
        
        self.play_btn = ctk.CTkButton(btn_row, text="▶ Play", command=self.toggle_playback, fg_color="#27ae60", hover_color="#229954", width=90, corner_radius=10, font=("Inter", 13))
        self.play_btn.pack(side="left", padx=2)
        
        self.stop_btn = ctk.CTkButton(btn_row, text="⏹ Stop", command=self.stop_video, fg_color="#e74c3c", hover_color="#c0392b", width=90, corner_radius=10, font=("Inter", 13))
        self.stop_btn.pack(side="left", padx=2)
        
        #seek slider
        self.seek_var = tk.IntVar(value=0)
        self.seek_slider = ctk.CTkSlider(playback_frame, from_=0, to=100, variable=self.seek_var, command=self.on_seek, progress_color="#3498db", button_color="#2980b9")
        self.seek_slider.pack(fill="x", pady=5, padx=10)
        
        self.time_label = ctk.CTkLabel(playback_frame, text="00:00 / 00:00", font=("Inter", 11), text_color="#7f8c8d")
        self.time_label.pack(anchor="w", padx=10)
        
        #loop checkbox
        self.loop_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(playback_frame, text="Loop Video", variable=self.loop_var, fg_color="#3498db", hover_color="#2980b9", font=("Inter", 12)).pack(anchor="w", pady=5, padx=10)
        
        #speed control
        speed_row = ctk.CTkFrame(playback_frame, fg_color="transparent")
        speed_row.pack(fill="x", pady=(5, 5), padx=10)
        ctk.CTkLabel(speed_row, text="Speed:", font=("Inter", 11), text_color="#7f8c8d").pack(side="left")
        self.speed_var = ctk.StringVar(value="1.0")
        speed_menu = ctk.CTkOptionMenu(speed_row, variable=self.speed_var, values=["0.25", "0.5", "0.75", "1.0", "1.5", "2.0"], width=80, fg_color="#3498db", button_color="#2980b9", font=("Inter", 12))
        speed_menu.pack(side="left", padx=5)
        
        #display mode control
        display_row = ctk.CTkFrame(playback_frame, fg_color="transparent")
        display_row.pack(fill="x", pady=(5, 5), padx=10)
        ctk.CTkLabel(display_row, text="Display:", font=("Inter", 11), text_color="#7f8c8d").pack(side="left")
        self.display_mode_var = ctk.StringVar(value="Fit")
        display_menu = ctk.CTkOptionMenu(display_row, variable=self.display_mode_var, values=["Fit", "Fill"], command=self.on_display_mode_change, width=80, fg_color="#9b59b6", button_color="#8e44ad", font=("Inter", 12))
        display_menu.pack(side="left", padx=5)
        
        #analysis mode control (performance vs accuracy)
        analysis_row = ctk.CTkFrame(playback_frame, fg_color="transparent")
        analysis_row.pack(fill="x", pady=(5, 10), padx=10)
        ctk.CTkLabel(analysis_row, text="Analysis:", font=("Inter", 11), text_color="#7f8c8d").pack(side="left")
        self.analysis_mode_var = ctk.StringVar(value="Balanced")
        analysis_menu = ctk.CTkOptionMenu(analysis_row, variable=self.analysis_mode_var, values=["Fast", "Balanced", "Detailed"], command=self.on_analysis_mode_change, width=90, fg_color="#e67e22", button_color="#d35400", font=("Inter", 12))
        analysis_menu.pack(side="left", padx=5)

        ctk.CTkFrame(self.controls_panel, height=2, fg_color="#bdc3c7").pack(fill="x", pady=10, padx=15)
        
        #form selection
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
            corner_radius=10,
            font=("Inter", 14)
        )
        self.form_button.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkFrame(self.controls_panel, height=2, fg_color="#bdc3c7").pack(fill="x", pady=10, padx=15)
        
        #status section
        self.status_label = ctk.CTkLabel(self.controls_panel, text="Status: Load a video", font=("Inter", 14), wraplength=220, text_color="#2c3e50")
        self.status_label.pack(fill="x", pady=5, anchor="w", padx=10)
        
        #confidence display
        confidence_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        confidence_frame.pack(fill="x", pady=5, padx=10)
        
        ctk.CTkLabel(confidence_frame, text="Confidence:", font=("Inter", 11), text_color="#7f8c8d").pack(side="left", padx=(0, 5))
        
        self.confidence_progress = ctk.CTkProgressBar(confidence_frame, mode='determinate', progress_color="#27ae60", height=10, width=120)
        self.confidence_progress.pack(side="left", fill="x", expand=True)
        self.confidence_progress.set(0)
        
        self.confidence_label = ctk.CTkLabel(confidence_frame, text="0%", font=("Inter", 11, "bold"), text_color="#27ae60")
        self.confidence_label.pack(side="left", padx=(5, 0))

        #keyboard shortcuts
        self.window.bind('<space>', lambda e: self.toggle_playback())
        self.window.bind('<Left>', lambda e: self.step_frame(-1))
        self.window.bind('<Right>', lambda e: self.step_frame(1))
        self.window.bind('<Control-o>', lambda e: self.open_video_dialog())

        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()
        
        #set window size and center
        width = int(self.screen_width * 0.8)
        height = int(self.screen_height * 0.8)
        x = (self.screen_width // 2) - (width // 2)
        y = (self.screen_height // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
        self.window.mainloop()
    
    def open_video_dialog(self):
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if path:
            self.load_video(path)
    
    def load_video(self, path):
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.video_name_label.configure(text="Error: Could not open video")
            self.status_label.configure(text="Status: Video load failed")
            return
        
        self.video_path = path
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.is_video_loaded = True
        self.is_playing = False
        
        video_name = os.path.basename(path)
        self.video_name_label.configure(text=video_name)
        self.seek_slider.configure(to=max(1, self.total_frames - 1))
        self.update_time_label()
        self.status_label.configure(text=f"Status: Loaded ({self.total_frames} frames)")
        self.play_btn.configure(text="▶ Play")
        
        print(f"[INFO] loaded video: {path} ({self.total_frames} frames @ {self.video_fps:.1f} fps)")
    
    def toggle_playback(self):
        if not self.is_video_loaded:
            return
        self.is_playing = not self.is_playing
        self.play_btn.configure(text="⏸ Pause" if self.is_playing else "▶ Play")
    
    def stop_video(self):
        self.is_playing = False
        self.current_frame_idx = 0
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.seek_var.set(0)
        self.play_btn.configure(text="▶ Play")
        self.update_time_label()
    
    def step_frame(self, delta):
        if not self.is_video_loaded:
            return
        self.is_playing = False
        self.play_btn.configure(text="▶ Play")
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_idx + delta))
        self.current_frame_idx = new_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
        self.seek_var.set(new_idx)
        self.update_time_label()
    
    def on_seek(self, value):
        if not self.is_video_loaded:
            return
        new_idx = int(float(value))
        if abs(new_idx - self.current_frame_idx) > 1:
            self.current_frame_idx = new_idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
            self.update_time_label()
    
    def update_time_label(self):
        if not self.is_video_loaded:
            self.time_label.configure(text="00:00 / 00:00")
            return
        current_sec = self.current_frame_idx / self.video_fps
        total_sec = self.total_frames / self.video_fps
        cur_min, cur_sec = divmod(int(current_sec), 60)
        tot_min, tot_sec = divmod(int(total_sec), 60)
        self.time_label.configure(text=f"{cur_min:02d}:{cur_sec:02d} / {tot_min:02d}:{tot_sec:02d}")

    def draw_text_with_bg(self, img, text, pos, font_face, font_scale, text_color, bg_color, thickness):
        (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        top_left = (pos[0], pos[1] - text_h - baseline)
        bottom_right = (pos[0] + text_w, pos[1] + baseline)
        cv2.rectangle(img, top_left, bottom_right, bg_color, cv2.FILLED)
        cv2.putText(img, text, (pos[0], pos[1]), font_face, font_scale, text_color, thickness)

    def resize_and_pad(self, img, size, pad_color=0):
        #fit mode - letterbox/pillarbox to preserve aspect ratio
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
    
    def resize_and_crop(self, img, size):
        #fill mode - crop to fill canvas while preserving aspect ratio
        h, w, _ = img.shape; sw, sh = size
        if w == 0 or h == 0 or sw == 0 or sh == 0: return np.zeros((sh, sw, 3), dtype=np.uint8)
        aspect = w / h; canvas_aspect = sw / sh
        if aspect > canvas_aspect:
            #image is wider - scale to match height, crop width
            new_h = sh; new_w = int(new_h * aspect)
        else:
            #image is taller - scale to match width, crop height
            new_w = sw; new_h = int(new_w / aspect)
        interp = cv2.INTER_AREA if new_w < w or new_h < h else cv2.INTER_LINEAR
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        #center crop
        start_x = (new_w - sw) // 2
        start_y = (new_h - sh) // 2
        cropped = scaled_img[start_y:start_y+sh, start_x:start_x+sw]
        return cropped
    
    def resize_preserve_aspect(self, img, target_width, target_height):
        #resize for processing while preserving aspect ratio
        h, w = img.shape[:2]
        if w == 0 or h == 0: return img
        aspect = w / h
        if aspect > target_width / target_height:
            new_w = target_width
            new_h = int(new_w / aspect)
        else:
            new_h = target_height
            new_w = int(new_h * aspect)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def on_display_mode_change(self, value):
        self.display_mode = value.lower()
    
    def on_analysis_mode_change(self, value):
        #adjust processing intervals based on analysis mode
        self.analysis_mode = value.lower()
        if value == "Fast":
            self.ml_inference_interval = 12
            self.stick_detection_interval = 8
        elif value == "Detailed":
            self.ml_inference_interval = 4
            self.stick_detection_interval = 2
        else:  #balanced
            self.ml_inference_interval = 8
            self.stick_detection_interval = 4
        print(f"[INFO] analysis mode: {value} (ML every {self.ml_inference_interval}, stick every {self.stick_detection_interval})")
    
    def video_loop(self):
        COLOR_DEFAULT = (255, 0, 0); COLOR_CORRECT = (0, 255, 0); COLOR_ERROR = (0, 0, 255)
        COLOR_PROMPT = (0, 255, 255); COLOR_WHITE = (255, 255, 255); COLOR_BLACK = (0, 0, 0)

        last_frame_time = time.time()
        
        while self.is_running:
            #no video loaded - show placeholder
            if not self.is_video_loaded or self.cap is None:
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    placeholder = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    placeholder[:] = (40, 40, 40)
                    cv2.putText(placeholder, "Load a video file to begin", 
                        (canvas_width//2 - 180, canvas_height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                    cv2.putText(placeholder, "Press Ctrl+O or click 'Open Video'", 
                        (canvas_width//2 - 200, canvas_height//2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                    if self.queue.full():
                        try: self.queue.get_nowait()
                        except queue.Empty: pass
                    self.queue.put(placeholder)
                time.sleep(0.1)
                continue
            
            #respect playback speed
            try:
                speed = float(self.speed_var.get())
            except:
                speed = 1.0
            frame_delay = 1.0 / (self.video_fps * speed)
            
            if self.is_playing:
                elapsed = time.time() - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(0.001)
                    continue
                last_frame_time = time.time()
            
            #read frame
            ret, frame = self.cap.read()
            
            if not ret:
                #end of video
                if self.loop_var.get():
                    self.current_frame_idx = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.is_playing = False
                    self.play_btn.configure(text="▶ Play")
                    time.sleep(0.05)
                    continue
            
            if self.is_playing:
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.seek_var.set(self.current_frame_idx)
                self.update_time_label()
            
            frame = cv2.flip(frame, 1)
            #preserve aspect ratio during processing resize - use lower res for performance
            processing_frame = self.resize_preserve_aspect(frame, 480, 360)
            
            #skip heavy processing on some frames for performance
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

            #update confidence display
            if self.last_known_results:
                result = self.last_known_results[0]
                predicted_class = result['predicted_class']
                predicted_class = re.sub(r'^\d+\.\s*', '', predicted_class)
                confidence = result['confidence']
                
                self.confidence_progress.set(confidence)
                self.confidence_label.configure(text=f"{int(confidence * 100)}%")
                
                if confidence > 0.60:
                    self.confidence_progress.configure(progress_color="#27ae60")
                    self.confidence_label.configure(text_color="#27ae60")
                elif confidence > 0.40:
                    self.confidence_progress.configure(progress_color="#f39c12")
                    self.confidence_label.configure(text_color="#f39c12")
                else:
                    self.confidence_progress.configure(progress_color="#e74c3c")
                    self.confidence_label.configure(text_color="#e74c3c")
            
            if self.last_known_results:
                result = self.last_known_results[0]
                x1, y1, x2, y2 = result['bbox']
                person_id = result['id']
                
                draw_color = COLOR_ERROR; box_color = COLOR_DEFAULT; is_correct = False

                if self.target_form:
                    feedback = self.feedback_analyzer.analyze(result, self.target_form)
                    is_correct = feedback['is_correct']
                    
                    if is_correct:
                        draw_color = COLOR_CORRECT
                        box_color = COLOR_CORRECT
                        current_state = 'correct'
                    else:
                        current_state = 'incorrect'
                    
                    if current_state != self.last_pose_state:
                        if self.last_pose_state is not None and self.state_frame_count >= self.min_state_frames:
                            if current_state == 'correct':
                                print(f"[ATTEMPT] correct (from {self.last_pose_state})")
                            elif self.last_pose_state == 'correct':
                                print(f"[ATTEMPT] incorrect (from correct)")
                        self.last_pose_state = current_state
                        self.state_frame_count = 1
                    else:
                        self.state_frame_count += 1
                
                cv2.rectangle(processing_frame, (x1, y1), (x2, y2), box_color, 2)
                
                if result['stick_endpoints']:
                    self.analyzer.draw_stick_debug(processing_frame, result['stick_endpoints'])

                if result.get('landmarks_absolute'):
                    landmarks_abs = result['landmarks_absolute']
                    
                    for idx, (lx, ly, lz) in enumerate(landmarks_abs):
                        cv2.circle(processing_frame, (lx, ly), 2, draw_color, -1)
                    
                    pose_connections = self.analyzer.mp_pose.POSE_CONNECTIONS
                    for connection in pose_connections:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks_abs) and end_idx < len(landmarks_abs):
                            start_pt = (int(landmarks_abs[start_idx][0]), int(landmarks_abs[start_idx][1]))
                            end_pt = (int(landmarks_abs[end_idx][0]), int(landmarks_abs[end_idx][1]))
                            cv2.line(processing_frame, start_pt, end_pt, draw_color, 2)

                if self.target_form:
                    #use feedback analyzer to get detailed feedback
                    feedback = self.feedback_analyzer.analyze(result, self.target_form)
                    prioritized_messages = self.feedback_analyzer.get_prioritized_messages(feedback, max_messages=3)
                    
                    #opencv feedback rendering - compact size (same as main app)
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
            
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            if self.display_mode == "fill":
                final_frame = self.resize_and_crop(processing_frame, size=(canvas_width, canvas_height))
            else:
                final_frame = self.resize_and_pad(processing_frame, size=(canvas_width, canvas_height))
            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(final_frame)
            
            self.frame_counter += 1
            
            #when paused, sleep more to reduce cpu usage
            if not self.is_playing:
                time.sleep(0.05)

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
        self.status_label.configure(text=f"Status: Analyzing '{pretty_name}'")
        print(f"[INFO] targeting: '{self.target_form}'")
    
    def on_closing(self):
        print("[INFO] closing...")
        self.is_running = False
        time.sleep(0.5)
        self.analyzer.close()
        if self.cap is not None:
            self.cap.release()
        self.window.destroy()
    
    def reset_feedback(self):
        self.target_form = None
        self.selected_form.set("Choose Arnis Form")
        self.status_label.configure(text="Status: Load a video")

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shell32.SetCurrentProcessExplicitAppUserModelID('TuroArnis.VideoTest.1.0')
    except:
        pass
    
    root = ctk.CTk()
    
    try:
        from app.utils.resource_path import get_resource_path
        icon_path = get_resource_path('app/assets/TA.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception as e:
        print(f"[WARNING] Could not set icon: {e}")
    
    app = TuroArnisVideoGUI(root, "TuroArnis - Arnis Form Correction (Video Test)")
