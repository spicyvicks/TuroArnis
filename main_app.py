# file: main_app.py

import sys
import cv2
import threading
import time
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import queue


from gui.results_window import ResultsWindow
from computer_vision.pose_analyzer import PoseAnalyzer

class TuroArnisGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # initialize our new pose analyzer class
        self.analyzer = PoseAnalyzer()

        # Set desired height and calculate width based on 16:9 aspect ratio
        target_height = 1920
        aspect_ratio = 16/9
        target_width = int(target_height * aspect_ratio)

        self.cap = cv2.VideoCapture(0)
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        
        # Get actual supported resolution (might be different from requested)
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set window size
        self.window.geometry(f"{self.cam_width}x{self.cam_height}")

        # Center the window on screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - self.cam_width) // 2
        y = (screen_height - self.cam_height) // 2
        self.window.geometry(f"+{x}+{y}")

        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        self.current_user = "Default User"

        # --- create and place gui widgets (this part is largely the same) ---
        self.video_label = ttk.Label(self.window)
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.controls_panel = ttk.Frame(self.window, padding=15, bootstyle="dark")
        self.controls_panel.place(x=20, y=20)

        ttk.Label(self.controls_panel, text="Controls", font=("-size 14 -weight bold"), bootstyle="inverse-dark").pack(pady=(0, 10), anchor=W)
        
        # user dropdown
        self.user_button = ttk.Menubutton(self.controls_panel, text=self.current_user, bootstyle="secondary")
        self.user_button.pack(fill=X, pady=5)
        self.user_menu = ttk.Menu(self.user_button)
        users = ["Default User", "John Doe", "Jane Smith"]
        for user_text in users:
            self.user_menu.add_command(label=user_text, command=lambda u=user_text: self.on_user_selected(u))
        self.user_button["menu"] = self.user_menu

        # form dropdown
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

        # feedback labels (simplified as feedback is on-screen now)
        ttk.Separator(self.controls_panel, orient=HORIZONTAL).pack(fill=X, pady=15)
        self.status_label = ttk.Label(self.controls_panel, text="Status: Select a form", font="-size 12", wraplength=220, bootstyle="inverse-dark")
        self.status_label.pack(fill=X, pady=5, anchor=W)

        self.view_all_results_button = ttk.Button(self.controls_panel, text="View All Results", command=self.open_results_window, bootstyle="info")
        self.view_all_results_button.pack(fill=X, pady=10, side=BOTTOM)

        # thread control and startup
        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()
        self.window.mainloop()

    def video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)

            # call the analyzer to do all the heavy lifting
            processed_frame = self.analyzer.process_frame(frame, self.target_form)

            # put the final, annotated frame in the queue
            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            
            self.queue.put(processed_frame)

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
            self.window.after(20, self.process_queue)

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
        time.sleep(0.5) # give the thread time to finish its current loop
        self.analyzer.close()
        self.cap.release()
        self.window.destroy()
        
    def on_user_selected(self, username):
        # this is now less critical but can be used for logging
        self.current_user = username
        self.user_button.config(text=username)
        print(f"current user set to: {username}")
    
    def reset_feedback(self):
        self.target_form = None
        self.form_button.config(text="Choose Arnis Form")
        self.status_label.config(text="Status: Select a form")

if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = TuroArnisGUI(root, "TuroArnis - Multi-User Form Corrector")