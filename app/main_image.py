import sys
import os
import cv2
import threading
import time
import re
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import queue
import numpy as np

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

#image testing config - uses resource path for deployment
TEST_IMAGE_PATH = get_resource_path('Left Temple Block.jpg')
DEFAULT_TEST_POSE_PRETTY_NAME = "Left Temple Block"

class TuroArnisGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        #set app icon for taskbar
        icon_path = get_resource_path('assets/TA.ico')
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
        
        self.window.update_idletasks()
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        
        #test user, no db
        self.current_user = {'id': 0, 'name': 'Test User (Image Mode)'}
        self.current_session_id = None

        self.frame_counter = 0
        self.processing_interval = 3
        self.last_known_results = []
        
        #state tracking
        self.last_pose_state = None
        self.state_frame_count = 0
        self.min_state_frames = 15

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
        
        #load test image
        self.static_image_original = cv2.imread(TEST_IMAGE_PATH)
        if self.static_image_original is None:
            print(f"[ERROR] could not load image: {TEST_IMAGE_PATH}")
            sys.exit(1)
        self.cap = None
        
        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=0)
        self.window.grid_columnconfigure(1, weight=1) 

        self.video_canvas = ttk.Canvas(self.window, background='black')
        self.video_canvas.grid(row=0, column=1, sticky="nsew")
        self.video_canvas.bind('<Configure>', self.on_canvas_resize)
        self.tk_image = None

        self.controls_panel = ttk.Frame(self.window, padding=15, bootstyle="light", width=250)
        self.controls_panel.grid(row=0, column=0, sticky="nsew")
        self.controls_panel.grid_propagate(False) 
        
        ttk.Label(self.controls_panel, text="Controls", font=("Inter", 18, "bold"), bootstyle="dark").pack(pady=(0, 10), anchor=W)

        user_frame = ttk.Labelframe(self.controls_panel, text="Current User", padding=10)
        user_frame.pack(fill=X, pady=5)
        ttk.Label(user_frame, text=self.current_user['name'], font=("Inter", 16, "bold"), bootstyle="info").pack(anchor=W)
        ttk.Label(user_frame, text="Image Testing Mode", font=("Inter", 12), bootstyle="secondary").pack(anchor=W)

        session_frame = ttk.Labelframe(self.controls_panel, text="Session", padding=10)
        session_frame.pack(fill=X, pady=5)
        
        self.session_status_label = ttk.Label(session_frame, text="Image Testing - No Sessions", font=("Inter", 12), bootstyle="secondary")
        self.session_status_label.pack(anchor=W, pady=2)
        
        ttk.Separator(self.controls_panel, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        self.practice_stances = {
            "Crown Thrust": "crown_thrust_correct", "Left Chest Thrust": "left_chest_thrust_correct",
            "Left Elbow Block": "left_elbow_block_correct", "Left Eye Thrust": "left_eye_thrust_correct",
            "Left Knee Block": "left_knee_block_correct", "Left Temple Block": "left_temple_block_correct",
            "Right Chest Thrust": "right_chest_thrust_correct", "Right Elbow Block": "right_elbow_block_correct",
            "Right Eye Thrust": "right_eye_thrust_correct", "Right Knee Block": "right_knee_block_correct",
            "Right Temple Block": "right_temple_block_correct", "Solar Plexus Thrust": "solar_plexus_thrust_correct"
        }
        self.form_button = ttk.Menubutton(self.controls_panel, text="Choose Arnis Form", bootstyle="primary")
        self.form_button.pack(fill=X, pady=5)
        self.form_menu = ttk.Menu(self.form_button)
        for pretty_name in self.practice_stances.keys():
            self.form_menu.add_command(label=pretty_name, command=lambda p=pretty_name: self.on_action_selected(p))
        self.form_button["menu"] = self.form_menu
        
        ttk.Separator(self.controls_panel, orient=HORIZONTAL).pack(fill=X, pady=15)
        self.status_label = ttk.Label(self.controls_panel, text="Status: Select a form", font=("Inter", 14), wraplength=220, bootstyle="dark")
        self.status_label.pack(fill=X, pady=5, anchor=W)
        
        self.keras_status_label = ttk.Label(self.controls_panel, text="Keras: N/A (0.00)", font=("Inter", 12), bootstyle="warning")
        self.keras_status_label.pack(fill=X, pady=5, anchor=W)
        
        self.view_all_results_button = ttk.Button(self.controls_panel, text="View All Results", command=self.open_results_window, bootstyle="info")
        self.view_all_results_button.pack(fill=X, pady=10, side=BOTTOM)

        #auto-select default pose
        if DEFAULT_TEST_POSE_PRETTY_NAME in self.practice_stances:
            self.target_form = self.practice_stances[DEFAULT_TEST_POSE_PRETTY_NAME]
            self.form_button.config(text=DEFAULT_TEST_POSE_PRETTY_NAME)
            self.status_label.config(text=f"Status: Analyzing '{DEFAULT_TEST_POSE_PRETTY_NAME}' (Image Test)")
            print(f"[INFO] targeting: '{self.target_form}'")

        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()
        
        #set window size and center it (must be done together)
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
            frame = self.static_image_original.copy()
            
            frame = cv2.flip(frame, 1)
            processing_frame = cv2.resize(frame, (640, 480))
            
            analysis_results = self.analyzer.process_frame(processing_frame)
            if analysis_results:
                self.last_known_results = analysis_results
                if analysis_results and len(analysis_results) > 0:
                    result = analysis_results[0]
                    print(f"[DEBUG] stick_endpoints: {result.get('stick_endpoints')}")

            feedback_x = processing_frame.shape[1] - 270; feedback_y = 30
            
            keras_status_text = "Keras: N/A (0.00)"
            if self.last_known_results:
                result = self.last_known_results[0]
                predicted_class = result['predicted_class']
                predicted_class = re.sub(r'^\d+\.\s*', '', predicted_class)
                confidence = result['confidence']
                pretty_class_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
                keras_status_text = f"Keras: {pretty_class_name} ({confidence:.2f})"
                if confidence > 0.60: self.keras_status_label.config(bootstyle="success")
                elif confidence > 0.40: self.keras_status_label.config(bootstyle="warning")
                else: self.keras_status_label.config(bootstyle="danger")
            self.keras_status_label.config(text=keras_status_text)
            
            if self.last_known_results:
                result = self.last_known_results[0]
                x1, y1, x2, y2 = result['bbox']
                person_id = result['id']
                
                draw_color = COLOR_ERROR; box_color = COLOR_DEFAULT; is_correct = False

                if self.target_form:
                    #use feedback analyzer to determine correctness
                    feedback = self.feedback_analyzer.analyze(result, self.target_form)
                    is_correct = feedback['is_correct']
                    
                    if is_correct:
                        draw_color = COLOR_CORRECT
                        box_color = COLOR_CORRECT
                        current_state = 'correct'
                    else:
                        current_state = 'incorrect'
                    
                    #state transition tracking (logs only)
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
                
                #use debug overlay for stick in test mode
                if result['stick_endpoints']:
                    self.analyzer.draw_stick_debug(processing_frame, result['stick_endpoints'])

                user_display_name = self.current_user['name'] if self.current_user else f"Person {person_id}"
                self.draw_text_with_bg(img=processing_frame, text=user_display_name, pos=(x1, y1 - 10), font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, text_color=COLOR_BLACK, bg_color=COLOR_WHITE, thickness=2)

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
                    prioritized_messages = self.feedback_analyzer.get_prioritized_messages(feedback, max_messages=4)
                    
                    #enhanced feedback ui with better styling
                    box_width = 420  #larger for 640x480 image
                    box_height = 220
                    box_x = processing_frame.shape[1] - box_width - 20
                    box_y = 20
                    
                    #draw shadow for depth
                    shadow_offset = 5
                    cv2.rectangle(processing_frame, 
                        (box_x + shadow_offset, box_y + shadow_offset), 
                        (box_x + box_width + shadow_offset, box_y + box_height + shadow_offset), 
                        (20, 20, 20), -1)
                    
                    #draw main feedback box
                    overlay = processing_frame.copy()
                    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                        (40, 44, 52), -1)
                    alpha = 0.92
                    processing_frame = cv2.addWeighted(overlay, alpha, processing_frame, 1 - alpha, 0)
                    
                    #draw border
                    if feedback['is_correct']:
                        border_color = (0, 200, 0)
                    elif feedback['severity'] == 'critical':
                        border_color = (0, 0, 220)
                    else:
                        border_color = (52, 152, 219)
                    cv2.rectangle(processing_frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                        border_color, 3)
                    
                    #render content
                    content_x = box_x + 20
                    content_y = box_y + 35
                    
                    if feedback['is_correct']:
                        cv2.putText(processing_frame, "Perfect Form!", 
                            (content_x, content_y), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(processing_frame, "Maintain this position", 
                            (content_x, content_y + 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
                    else:
                        if prioritized_messages:
                            cv2.putText(processing_frame, "Form Feedback", 
                                (content_x, content_y), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            cv2.line(processing_frame, 
                                (content_x, content_y + 10), 
                                (box_x + box_width - 20, content_y + 10), 
                                (100, 100, 100), 1)
                            
                            msg_y = content_y + 45
                            for i, (message, msg_type) in enumerate(prioritized_messages):
                                if msg_y > box_y + box_height - 25:
                                    break
                                
                                if msg_type == 'error':
                                    icon = "X"
                                    msg_color = (0, 100, 255)
                                    icon_color = (0, 100, 255)
                                elif msg_type == 'warning':
                                    icon = "!"
                                    msg_color = (0, 165, 255)
                                    icon_color = (0, 165, 255)
                                else:
                                    icon = ">"
                                    msg_color = (200, 200, 0)
                                    icon_color = (200, 200, 0)
                                
                                cv2.putText(processing_frame, icon, 
                                    (content_x, msg_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, icon_color, 2, cv2.LINE_AA)
                                
                                max_chars = 42
                                display_msg = message[:max_chars] + "..." if len(message) > max_chars else message
                                cv2.putText(processing_frame, display_msg, 
                                    (content_x + 25, msg_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, msg_color, 1, cv2.LINE_AA)
                                
                                msg_y += 40
                        else:
                            pretty_form_name = self.form_button.cget('text')
                            if pretty_form_name != "Choose Arnis Form":
                                cv2.putText(processing_frame, "Adjust to:", 
                                    (content_x, content_y), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                                
                                words = pretty_form_name.split()
                                line1 = ' '.join(words[:3])
                                line2 = ' '.join(words[3:]) if len(words) > 3 else ""
                                
                                cv2.putText(processing_frame, line1, 
                                    (content_x, content_y + 45), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
                                if line2:
                                    cv2.putText(processing_frame, line2, 
                                        (content_x, content_y + 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            
            canvas_width = self.video_canvas.winfo_width(); canvas_height = self.video_canvas.winfo_height()
            final_frame = self.resize_and_pad(processing_frame, size=(canvas_width, canvas_height))
            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(final_frame)
            
            self.frame_counter += 1
            time.sleep(0.1)

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
        self.form_button.config(text=pretty_name)
        self.status_label.config(text=f"Status: Analyzing '{pretty_name}' (Image Test)")
        print(f"[INFO] targeting: '{self.target_form}'")
    
    def open_results_window(self):
        from ttkbootstrap.dialogs import Messagebox
        Messagebox.show_info("Image Testing Mode - No results database available", "Info")
    
    def on_closing(self):
        print("[INFO] closing...")
        self.is_running = False
        time.sleep(0.5)
        self.analyzer.close()
        self.window.destroy()
    
    def reset_feedback(self):
        self.target_form = None
        self.form_button.config(text="Choose Arnis Form")
        self.status_label.config(text="Status: Select a form")

if __name__ == "__main__":
    #windows: set app id so taskbar icon shows properly
    try:
        from ctypes import windll
        import os
        #set unique app id for windows taskbar
        windll.shell32.SetCurrentProcessExplicitAppUserModelID('TuroArnis.ImageTest.1.0')
    except:
        pass  #not on windows or failed
    
    root = ttk.Window(themename="flatly")
    
    #set icon before creating the gui
    try:
        from app.utils.resource_path import get_resource_path
        import os
        icon_path = get_resource_path('assets/TA.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception as e:
        print(f"[WARNING] Could not set icon: {e}")
    
    app = TuroArnisGUI(root, "TuroArnis - Arnis Form Correction (Image Test)")