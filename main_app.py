import sys
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


class TuroArnisGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.state('zoomed') 

        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.analyzer = PoseAnalyzer()
        self.cap = cv2.VideoCapture(0)

        self.frame_counter = 0
        self.processing_interval = 2 
        self.last_known_results = []

        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        self.current_user = "Default User"

        #widgets
        self.video_label = ttk.Label(self.window)
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.controls_panel = ttk.Frame(self.window, padding=15, bootstyle="dark")
        self.controls_panel.place(x=20, y=20)
        
        ttk.Label(self.controls_panel, text="Controls", font=("-size 14 -weight bold"), bootstyle="inverse-dark").pack(pady=(0, 10), anchor=W)
        self.user_button = ttk.Menubutton(self.controls_panel, text=self.current_user, bootstyle="secondary")
        self.user_button.pack(fill=X, pady=5)
        self.user_menu = ttk.Menu(self.user_button)
        users = ["Default User", "John Doe", "Jane Smith"]
        for user_text in users:
            self.user_menu.add_command(label=user_text, command=lambda u=user_text: self.on_user_selected(u))
        self.user_button["menu"] = self.user_menu
        
        self.practice_stances = {
            "Crown Thrust": "crown_thrust_correct", 
            "Left Chest Thrust": "left_chest_thrust_correct",
            "Left Elbow Block": "left_elbow_block_correct", 
            "Left Eye Thrust": "left_eye_thrust_correct",
            "Left Knee Block": "left_knee_block_correct", 
            "Left Temple Block": "left_temple_block_correct",
            "Right Chest Thrust": "right_chest_thrust_correct", 
            "Right Elbow Block": "right_elbow_block_correct",
            "Right Eye Thrust": "right_eye_thrust_correct", 
            "Right Knee Block": "right_knee_block_correct",
            "Right Temple Block": "right_temple_block_correct", 
            "Solar Plexus Thrust": "solar_plexus_thrust_correct"
        }
        self.form_button = ttk.Menubutton(self.controls_panel, text="Choose Arnis Form", bootstyle="primary")
        self.form_button.pack(fill=X, pady=5)
        self.form_menu = ttk.Menu(self.form_button)
        for pretty_name in self.practice_stances.keys():
            self.form_menu.add_command(label=pretty_name, command=lambda p=pretty_name: self.on_action_selected(p))
        self.form_button["menu"] = self.form_menu
        
        ttk.Separator(self.controls_panel, orient=HORIZONTAL).pack(fill=X, pady=15)
        self.status_label = ttk.Label(self.controls_panel, text="Status: Select a form", font="-size 12", wraplength=220, bootstyle="inverse-dark")
        self.status_label.pack(fill=X, pady=5, anchor=W)
        
        self.view_all_results_button = ttk.Button(self.controls_panel, text="View All Results", command=self.open_results_window, bootstyle="info")
        self.view_all_results_button.pack(fill=X, pady=10, side=BOTTOM)

        # startup
        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()
        self.window.mainloop()

    def resize_and_pad(self, img, size, pad_color=0):
        h, w, _ = img.shape
        sw, sh = size
        interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC
        aspect = w / h
        if aspect > sw / sh:
            new_w = sw
            new_h = np.round(new_w / aspect).astype(int)
            pad_vert = (sh - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        else:
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = (sw - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[pad_color]*3)
        return padded_img

    def video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            
            processing_frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()
            
            if self.frame_counter % (self.processing_interval + 1) == 0:
                self.last_known_results = self.analyzer.process_frame(processing_frame)

            if self.last_known_results:
                for result in self.last_known_results:
                    px1, py1, px2, py2 = result['bbox']
                    h_proc, w_proc, _ = processing_frame.shape
                    h_disp, w_disp, _ = display_frame.shape
                    
                    x1 = int(px1 * w_disp / w_proc)
                    y1 = int(py1 * h_disp / h_proc)
                    x2 = int(px2 * w_disp / w_proc)
                    y2 = int(py2 * h_disp / h_proc)
                    
                    person_id = result['id']
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display_frame, f"User {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

                    if result['landmarks']:
                        self.analyzer.mp_drawing.draw_landmarks(
                            display_frame, result['landmarks'], self.analyzer.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.analyzer.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                            connection_drawing_spec=self.analyzer.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )

                    if self.target_form:
                        predicted_class = result['predicted_class']
                        confidence = result['confidence']
                        live_angles = result['live_angles']
                        
                        if predicted_class == self.target_form and confidence > 0.60:
                            ideal_pose = POSE_LIBRARY.get(self.target_form)
                            error_messages = []
                            is_correct = True
                            
                            if ideal_pose and live_angles:
                                for joint, ideal_range in ideal_pose.items():
                                    live_angle = live_angles.get(joint)
                                    if live_angle is not None:
                                        min_angle, max_angle = ideal_range
                                        if not (min_angle <= live_angle <= max_angle):
                                            is_correct = False
                                            feedback = "too bent" if live_angle < min_angle else "too straight"
                                            error_messages.append(f"{joint.replace('_', ' ').title()} {feedback}")
                            
                            if is_correct:
                                cv2.putText(display_frame, "Correct!", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                for i, msg in enumerate(error_messages[:2]): #display 2 errors
                                    cv2.putText(display_frame, msg, (x1, y2 + 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            pretty_form_name = self.form_button.cget('text')
                            if pretty_form_name != "Choose Arnis Form":
                                cv2.putText(display_frame, f"Adjust to {pretty_form_name}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self.frame_counter += 1
            
            final_frame = self.resize_and_pad(display_frame, size=(self.screen_width, self.screen_height))

            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(final_frame)

    def process_queue(self):
        try:
            frame = self.queue.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        finally:
            self.window.after(30, self.process_queue)
    
    def on_action_selected(self, pretty_name):
        self.form_button.config(text=pretty_name)
        self.target_form = self.practice_stances[pretty_name]
        self.status_label.config(text=f"Status: Analyzing '{pretty_name}'")
        print(f"targeting model class: '{self.target_form}'")
    
    def open_results_window(self):
        ResultsWindow(self.window)
    
    def on_closing(self):
        print("closing application...")
        self.is_running = False
        time.sleep(0.5)
        self.analyzer.close()
        self.cap.release()
        self.window.destroy()
    
    def on_user_selected(self, username):
        self.current_user = username
        self.user_button.config(text=username)
        print(f"current user set to: {username}")
    
    def reset_feedback(self):
        self.target_form = None
        self.form_button.config(text="Choose Arnis Form")
        self.status_label.config(text="Status: Select a form")

if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = TuroArnisGUI(root, "TuroArnis - Arnis Form Correction")