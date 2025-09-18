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
import queue

from ultralytics import YOLO
from sort import Sort

# (resultswindow class remains exactly the same)
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
        
        # initialize models
        self.yolo_model = YOLO('yolov8n.pt') 
        self.tracker = Sort()
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

        # --- new: simplified and robust window sizing ---
        self.cap = cv2.VideoCapture(0)
        # get the default webcam resolution
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # set the window size to match the webcam
        self.window.geometry(f"{self.cam_width}x{self.cam_height}")


        # queue for thread-safe communication
        self.queue = queue.Queue(maxsize=1)

        # create and place widgets
        self.video_label = ttk.Label(self.window)
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.controls_panel = ttk.Frame(self.window, padding=15)
        
        self.controls_panel.configure(bootstyle="dark") 
        
        self.controls_panel.place(x=20, y=20)

        # populate controls panel
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
        
        self.remarks_label = ttk.Label(self.controls_panel, text="Remarks: Multi-user mode active.", font="-size 12", wraplength=220, bootstyle="inverse-dark")
        self.remarks_label.pack(fill=X, pady=5, anchor=W)
        self.accuracy_label = ttk.Label(self.controls_panel, text="Accuracy: N/A", font="-size 12", bootstyle="inverse-dark")
        self.accuracy_label.pack(fill=X, pady=5, anchor=W)
        
        self.view_all_results_button = ttk.Button(self.controls_panel, text="View All Results", command=self.open_results_window, bootstyle="info")
        self.view_all_results_button.pack(fill=X, pady=10, side=BOTTOM)

        # thread control and startup
        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue() 
        self.window.mainloop()

    def process_queue(self):
        try:
            frame_data = self.queue.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        finally:
            self.window.after(20, self.process_queue)

    # --- new helper function for drawing landmarks ---
    def draw_landmarks_on_main_frame(self, main_frame, pose_results, crop_x1, crop_y1):
        if pose_results.pose_landmarks:
            # create a copy of the landmark object to modify it
            translated_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
            translated_landmarks.landmark.extend(pose_results.pose_landmarks.landmark)

            # get the dimensions of the main frame
            frame_height, frame_width, _ = main_frame.shape
            
            # translate coordinates
            for landmark in translated_landmarks.landmark:
                # convert landmark from relative (0-1) on crop to absolute on crop
                pixel_x = landmark.x * (crop_x2 - crop_x1)
                pixel_y = landmark.y * (crop_y2 - crop_y1)
                
                # add the crop's offset
                pixel_x += crop_x1
                pixel_y += crop_y1
                
                # convert back to relative on the main frame
                landmark.x = pixel_x / frame_width
                landmark.y = pixel_y / frame_height

            # draw the translated landmarks
            self.mp_drawing.draw_landmarks(
                main_frame,
                translated_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

    def video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            
            # yolo detection
            results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0]) # filter for persons
            detections = np.empty((0, 5))

            for r in results_yolo:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
            
            tracked_persons = self.tracker.update(detections)

            for person in tracked_persons:
                x1, y1, x2, y2, person_id = map(int, person)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f"User {person_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                
                if self.target_form and self.model:
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0: continue

                    image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(image_rgb)

                    # --- corrected logic: draw landmarks back on the main frame ---
                    if pose_results.pose_landmarks:
                        self.draw_landmarks_on_main_frame(frame, pose_results, x1, y1)

                    if pose_results.pose_world_landmarks:
                        try:
                            landmarks = pose_results.pose_world_landmarks.landmark
                            row = []
                            for lm in landmarks: row.extend([lm.x, lm.y, lm.z])

                            x_live = pd.DataFrame([row])
                            prediction_index = self.model.predict(x_live)[0]
                            predicted_class = self.class_names[prediction_index]
                            
                            if predicted_class == self.target_form:
                                remarks_text = "Correct!"
                            else:
                                remarks_text = "Adjust Form"
                            
                            cv2.putText(frame, remarks_text, (x1, y2 + 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        except Exception as e:
                            print(f"error during prediction for user {person_id}: {e}")

            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            
            self.queue.put(frame)

    # (on_user_selected, on_action_selected, etc. are the same)
    def on_user_selected(self, username):
        self.current_user = username
        self.user_button.config(text=username)
        print(f"user changed to: {username}")
        self.reset_feedback()
    
    def on_action_selected(self, pretty_name):
        self.form_button.config(text=pretty_name)
        self.target_form = self.practice_stances[pretty_name]
    
        print(f"user selected '{pretty_name}', targeting model class: '{self.target_form}'")
    
    def reset_feedback(self):
        self.target_form = None
        self.form_button.config(text="Choose Arnis Form")
    
    def open_results_window(self):
        ResultsWindow(self.window)
    
    def on_closing(self):
        print("closing application...")
        self.is_running = False
        time.sleep(0.1) 
        if self.cap.isOpened():
            self.cap.release()
        self.pose.close()
    
        self.window.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="superhero") 
    app = TuroArnisGUI(root, "TuroArnis - Multi-User Form Corrector")