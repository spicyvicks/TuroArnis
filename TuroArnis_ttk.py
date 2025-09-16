# import necessary libraries
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import threading
import time
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

#ERRORR ON START WSADNAOIWNDOIAWNFOIANWFOIANFEI

class ResultsWindow(ttk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent, title="All User Results")
        self.geometry("1000x600")
        columns = ("user", "form", "remarks", "accuracy")
        self.table = ttk.Treeview(self, columns=columns, show='headings', bootstyle="primary")
        self.table.heading("user", text="User")
        self.table.heading("form", text="Form")
        self.table.heading("remarks", text="Remarks")
        self.table.heading("accuracy", text="Accuracy")
        self.table.column("user", width=100, anchor=W)
        self.table.column("form", width=200)
        self.table.column("remarks", width=400)
        self.table.column("accuracy", width=100, anchor=CENTER)
        self.table.pack(expand=True, fill=BOTH, padx=10, pady=10)
        self.populate_table()

    def populate_table(self):
        mock_data = [
            {"user": "John Doe", "form": "Left Temple Block", "remarks": "Excellent form!", "accuracy": "98%"},
            {"user": "Jane Smith", "form": "Right Eye Thrust", "remarks": "Good speed.", "accuracy": "95%"},
            {"user": "John Doe", "form": "Solar Plexus Thrust", "remarks": "Slightly off-balance.", "accuracy": "89%"},
            {"user": "Default User", "form": "Left Knee Block", "remarks": "Very precise.", "accuracy": "99%"},
            {"user": "Jane Smith", "form": "Crown Thrust", "remarks": "Correction needed on elbow angle.", "accuracy": "87%"}
        ]
        for row_data in mock_data:
            self.table.insert("", END, values=(row_data["user"], row_data["form"], row_data["remarks"], row_data["accuracy"]))

class TuroArnisGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1920x1080")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = None
        self.class_names = []
        self.target_form = None
        self.current_user = "Default User"
        try:
            self.model = joblib.load('arnis_random_forest_classifier.joblib')
            self.class_names = joblib.load('arnis_class_names.joblib')
            print("[info] model and class names loaded successfully.")
        except FileNotFoundError:
            print("[error] classifier model or class names file not found.")

        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.video_label = ttk.Label(self.window)
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.controls_panel = ttk.Frame(self.window, padding=15)
        
        self.controls_panel.configure(bootstyle="dark") 
        
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
        self.remarks_label = ttk.Label(self.controls_panel, text="Remarks: Select a form to begin.", font="-size 12", wraplength=220, bootstyle="inverse-dark")
        self.remarks_label.pack(fill=X, pady=5, anchor=W)
        self.accuracy_label = ttk.Label(self.controls_panel, text="Accuracy: N/A", font="-size 12", bootstyle="inverse-dark")
        self.accuracy_label.pack(fill=X, pady=5, anchor=W)
        
        self.view_all_results_button = ttk.Button(self.controls_panel, text="View All Results", command=self.open_results_window, bootstyle="info")
        self.view_all_results_button.pack(fill=X, pady=10, side=BOTTOM)

        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()


    def on_user_selected(self, username):
        self.current_user = username
        self.user_button.config(text=username)
        print(f"user changed to: {username}")
        self.reset_feedback()

    def on_action_selected(self, pretty_name):
        self.form_button.config(text=pretty_name)
        self.target_form = self.practice_stances[pretty_name]
        self.remarks_label.config(text=f"remarks: now perform {pretty_name}", bootstyle="inverse-dark")
        self.accuracy_label.config(text="accuracy: n/a")
        print(f"user selected '{pretty_name}', targeting model class: '{self.target_form}'")
        
    def reset_feedback(self):
        self.target_form = None
        self.form_button.config(text="choose arnis form")
        self.remarks_label.config(text="remarks: select a form to begin.", bootstyle="inverse-dark")
        self.accuracy_label.config(text="accuracy: n/a")

    def open_results_window(self):
        ResultsWindow(self.window)
        
    def on_closing(self):
        print("closing application...")
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        self.pose.close()
        self.window.destroy()
        
    def video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.resize(frame, (1920, 1080))
            
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            frame_for_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame_for_display, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            if self.target_form and self.model and results.pose_world_landmarks:
                try:
                    landmarks = results.pose_world_landmarks.landmark
                    row = []
                    for lm in landmarks: row.extend([lm.x, lm.y, lm.z])
                    
                    X_live = pd.DataFrame([row])
                    prediction_index = self.model.predict(X_live)[0]
                    predicted_class = self.class_names[prediction_index]
                    prediction_proba = self.model.predict_proba(X_live)[0]
                    confidence = prediction_proba[prediction_index]

                    if predicted_class == self.target_form and confidence > 0.60:
                        self.remarks_label.config(text="remarks: correct!", bootstyle="success")
                        self.accuracy_label.config(text=f"accuracy: {confidence:.2%}")
                    else:
                        self.remarks_label.config(text="remarks: adjust for your chosen form", bootstyle="danger")
                        self.accuracy_label.config(text="accuracy: n/a")

                except Exception as e:
                    print(f"error during prediction: {e}")
                    self.remarks_label.config(text="remarks: error", bootstyle="danger")
                    
            elif self.target_form and not results.pose_world_landmarks:
                 self.remarks_label.config(text="remarks: no pose detected.", bootstyle="inverse-dark")

            img = Image.fromarray(cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            time.sleep(0.01)

if __name__ == "__main__":
    root = ttk.Window(themename="superhero") 
    app = TuroArnisGUI(root, "TuroArnis - Arnis Form Corrector")