import sys
import os
import cv2
import threading
import time
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import queue
import numpy as np

from gui.results_window import ResultsWindow
from computer_vision.pose_analyzer import PoseAnalyzer
from pose_definitions import POSE_LIBRARY

TEST_IMAGE_PATH = 'Left Temple Block.jpg' 
DEFAULT_TEST_POSE_PRETTY_NAME = "Left Temple Block" 

class TuroArnisGUI: 
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.frame_counter = 0
        self.processing_interval = 3
        self.last_known_results = []

        stick_model_path = 'runs/pose/arnis_stick_detector/weights/best.pt'
        self.analyzer = PoseAnalyzer(
            detection_interval=self.processing_interval,
            stick_model_path=stick_model_path if os.path.exists(stick_model_path) else None,
            debug_stick=True  
        )
        
        self.static_image_original = cv2.imread(TEST_IMAGE_PATH)
        if self.static_image_original is None:
            print(f"[CRITICAL ERROR] Could not load image at: {TEST_IMAGE_PATH}. Please check the path.")
            sys.exit(1)
        self.cap = None 
        
        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        self.current_user = "Default User"
        
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=0) 
        self.window.grid_columnconfigure(1, weight=1) 

        self.video_canvas = ttk.Canvas(self.window, background='black')
        self.video_canvas.grid(row=0, column=1, sticky="nsew")
        self.video_canvas.bind('<Configure>', self.on_canvas_resize)
        self.tk_image = None 

        self.controls_panel = ttk.Frame(self.window, padding=15, bootstyle="dark", width=250)
        self.controls_panel.grid(row=0, column=0, sticky="nsew")
        self.controls_panel.grid_propagate(False) 
        
        ttk.Label(self.controls_panel, text="Controls", font="Arial 14 bold", bootstyle="inverse-dark").pack(pady=(0, 10), anchor=W)
        self.user_button = ttk.Menubutton(self.controls_panel, text=self.current_user, bootstyle="secondary")
        self.user_button.pack(fill=X, pady=5)
        self.user_menu = ttk.Menu(self.user_button)
        users = ["Default User", "John Doe", "Jane Smith"]
        for user_text in users:
            self.user_menu.add_command(label=user_text, command=lambda u=user_text: self.on_user_selected(u))
        self.user_button["menu"] = self.user_menu
        
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
        self.status_label = ttk.Label(self.controls_panel, text="Status: Select a form", font="Arial 12", wraplength=220, bootstyle="inverse-dark")
        self.status_label.pack(fill=X, pady=5, anchor=W)
        
        self.keras_status_label = ttk.Label(self.controls_panel, text="Keras: N/A (0.00)", font="Arial 10", bootstyle="warning")
        self.keras_status_label.pack(fill=X, pady=5, anchor=W)
        
        self.feedback_label = ttk.Label(self.controls_panel, text="", font="Arial 9", wraplength=220, bootstyle="inverse-dark", justify=LEFT)
        self.feedback_label.pack(fill=X, pady=5, anchor=W)
        
        self.view_all_results_button = ttk.Button(self.controls_panel, text="View All Results", command=self.open_results_window, bootstyle="info")
        self.view_all_results_button.pack(fill=X, pady=10, side=BOTTOM)

        if DEFAULT_TEST_POSE_PRETTY_NAME in self.practice_stances:
            self.target_form = self.practice_stances[DEFAULT_TEST_POSE_PRETTY_NAME]
            self.form_button.config(text=DEFAULT_TEST_POSE_PRETTY_NAME)
            self.status_label.config(text=f"Status: Analyzing '{DEFAULT_TEST_POSE_PRETTY_NAME}' (Image Test)")
            print(f"[INFO] Targeting model class: '{self.target_form}'")

        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()
        
        self.window.geometry(f"{int(self.screen_width * 0.8)}x{int(self.screen_height * 0.8)}")
        self.window.mainloop()

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
            processing_frame = cv2.resize(frame, (640, 480))
            analysis_results = self.analyzer.process_frame(processing_frame)
            
            if analysis_results:
                self.last_known_results = analysis_results
                if analysis_results and len(analysis_results) > 0:
                    result = analysis_results[0]
                    print(f"[DEBUG-MAIN] Analysis result stick_endpoints: {result.get('stick_endpoints')}")

            keras_status_text = "Keras: N/A (0.00)"
            feedback_text = ""
            
            if self.last_known_results and len(self.last_known_results) > 0:
                result = self.last_known_results[0] 
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                pretty_class_name = predicted_class.replace('_correct', '').replace('_', ' ').title()
                keras_status_text = f"Keras: {pretty_class_name} ({confidence:.2f})"
                if confidence > 0.60: self.keras_status_label.config(bootstyle="success")
                elif confidence > 0.40: self.keras_status_label.config(bootstyle="warning")
                else: self.keras_status_label.config(bootstyle="danger")
            self.keras_status_label.config(text=keras_status_text)
            
            if self.last_known_results and len(self.last_known_results) > 0:
                result = self.last_known_results[0] 
                x1, y1, x2, y2 = result['bbox']
                person_id = result['id']
                
                draw_color = COLOR_DEFAULT
                box_color = COLOR_DEFAULT
                is_correct = False
                error_messages = []

                if self.target_form:
                    predicted_class = result['predicted_class']
                    confidence = result['confidence']
                    live_angles = result['live_angles']
                    
                    import re
                    predicted_normalized = re.sub(r'^\d+\.\s*', '', predicted_class.strip())
                    target_normalized = re.sub(r'^\d+\.\s*', '', self.target_form.strip())
                    
                    print(f"[DEBUG] Predicted: '{predicted_class}' -> '{predicted_normalized}'")
                    print(f"[DEBUG] Target: '{self.target_form}' -> '{target_normalized}'")
                    print(f"[DEBUG] Match: {predicted_normalized == target_normalized} | Confidence: {confidence:.2f}")
                    
                    if predicted_normalized == target_normalized and confidence > 0.60:
                        ideal_pose = POSE_LIBRARY.get(self.target_form)
                        pose_is_perfect = True
                        
                        if ideal_pose and live_angles:
                            for joint, ideal_range in ideal_pose.items():
                                live_angle = live_angles.get(joint)
                                if live_angle is not None:
                                    min_angle, max_angle = ideal_range
                                    if not (min_angle <= live_angle <= max_angle):
                                        pose_is_perfect = False
                                        feedback = "too bent" if live_angle < min_angle else "too straight"
                                        error_messages.append(f"{joint.replace('_', ' ').title()}: {feedback}")
                        else:
                            print(f"[DEBUG] No ideal pose in POSE_LIBRARY, trusting model")
                            pose_is_perfect = True
                        
                        if pose_is_perfect:
                            is_correct = True
                            draw_color = COLOR_CORRECT
                            feedback_text = "✓ Perfect Form!"
                            print(f"[DEBUG] Setting GREEN color")
                        else:
                            draw_color = COLOR_ERROR
                            feedback_text = "Adjustments:\n" + "\n".join(error_messages[:3])
                            print(f"[DEBUG] Setting RED color - {len(error_messages)} errors")
                    else:
                        draw_color = COLOR_ERROR
                        if confidence <= 0.60:
                            feedback_text = "Low confidence - adjust pose"
                        else:
                            feedback_text = f"Wrong pose detected"
                        print(f"[DEBUG] No match or low confidence - RED")
                else:
                    feedback_text = "Select a target form"
                
                self.feedback_label.config(text=feedback_text)
                print(f"[DEBUG] Feedback: {feedback_text}")
                
                cv2.rectangle(processing_frame, (x1, y1), (x2, y2), box_color, 2)
                
                print(f"[DEBUG-DRAW] Checking stick_endpoints: {result.get('stick_endpoints')}")
                if result['stick_endpoints']:
                    pt1, pt2 = result['stick_endpoints']
                    print(f"[DEBUG-DRAW] ✓ Drawing stick line from {pt1} to {pt2}")
                    cv2.line(processing_frame, pt1, pt2, COLOR_PROMPT, 4)
                else:
                    print(f"[DEBUG-DRAW] ✗ No stick to draw")

                self.draw_text_with_bg(img=processing_frame, text=f"User {person_id}", pos=(x1, y1 - 10), font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, text_color=COLOR_BLACK, bg_color=COLOR_WHITE, thickness=2)

                if result['landmarks']:
                    landmark_spec = self.analyzer.mp_drawing.DrawingSpec(color=draw_color, thickness=2, circle_radius=2)
                    connection_spec = self.analyzer.mp_drawing.DrawingSpec(color=draw_color, thickness=2, circle_radius=2)
                    self.analyzer.mp_drawing.draw_landmarks(processing_frame, result['landmarks'], self.analyzer.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=landmark_spec, connection_drawing_spec=connection_spec)
            
            canvas_width = self.video_canvas.winfo_width(); canvas_height = self.video_canvas.winfo_height()
            final_frame = self.resize_and_pad(processing_frame, size=(canvas_width, canvas_height))
            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(final_frame)
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
    
    def on_canvas_resize(self, event): self.process_queue() 

    def on_action_selected(self, pretty_name):
        self.target_form = self.practice_stances[pretty_name]
        self.form_button.config(text=pretty_name)
        self.status_label.config(text=f"Status: Analyzing '{pretty_name}'")
        print(f"[INFO] Targeting model class: '{self.target_form}'")
    
    def open_results_window(self): ResultsWindow(self.window)
    
    def on_closing(self):
        print("[INFO] Closing application...")
        self.is_running = False
        time.sleep(0.5)
        self.analyzer.close()
        self.window.destroy()
    
    def on_user_selected(self, username):
        self.current_user = username
        self.user_button.config(text=username)
        print(f"[INFO] Current user set to: {username}")
    
    def reset_feedback(self):
        self.target_form = None
        self.form_button.config(text="Choose Arnis Form")
        self.status_label.config(text="Status: Select a form")

if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = TuroArnisGUI(root, "TuroArnis - Arnis Form Correction (Image Test)")