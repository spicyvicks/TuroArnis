import sys
import cv2
import random
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QMenu, QTableWidget,
                             QTableWidgetItem, QHeaderView)


class ResultsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("All User Results")
        self.setStyleSheet("background-color: white;")

        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["User", "Form", "Remarks", "Accuracy", "Action"])
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget { border: none; gridline-color: transparent; }
            QHeaderView::section { background-color: #f0f0f0; padding: 6px; border: none; border-bottom: 1px solid #ddd; font-weight: bold; }
            QTableWidget::item { padding-left: 10px; padding-right: 10px; }
        """)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        
        self.populate_table()

    def populate_table(self):
        mock_data = [
            {"user": "John Doe", "form": "Left Temple Block", "remarks": "Excellent form!", "accuracy": "98%"},
            {"user": "Jane Smith", "form": "Right Eye Thrust", "remarks": "Good speed.", "accuracy": "95%"},
            {"user": "John Doe", "form": "Solar Plexus Thrust", "remarks": "Slightly off-balance.", "accuracy": "89%"},
            {"user": "Default User", "form": "Left Knee Block", "remarks": "Very precise.", "accuracy": "99%"},
            {"user": "Jane Smith", "form": "Crown Thrust", "remarks": "Correction needed on elbow angle.", "accuracy": "87%"}
        ]
        self.table.setRowCount(len(mock_data))
        for row_index, row_data in enumerate(mock_data):
            self.table.setItem(row_index, 0, QTableWidgetItem(row_data["user"]))
            self.table.setItem(row_index, 1, QTableWidgetItem(row_data["form"]))
            self.table.setItem(row_index, 2, QTableWidgetItem(row_data["remarks"]))
            accuracy_item = QTableWidgetItem(row_data["accuracy"])
            accuracy_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_index, 3, accuracy_item)
            view_button = QPushButton("View")
            view_button.clicked.connect(lambda checked, r=row_index: self.view_details(r))
            self.table.setCellWidget(row_index, 4, view_button)

    def view_details(self, row):
        user = self.table.item(row, 0).text()
        form = self.table.item(row, 1).text()
        print(f"Viewing details for row {row}: User '{user}', Form '{form}'")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TuroArnis")
        self.results_win = None

        self.model = None
        self.class_names = []
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.target_form = None

        try:
            self.model = joblib.load('arnis_random_forest_classifier.joblib')
            self.class_names = joblib.load('arnis_class_names.joblib')
            print("[INFO] Model and class names loaded successfully.")
        except FileNotFoundError:
            print("[ERROR] Model or class names file not found.")
            
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from video capture.")
            sys.exit()
        height, width, _ = frame.shape
        self.setFixedSize(QSize(width, height))
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.video_label)
        
        self.main_controls_panel = QWidget(self)
        self.main_controls_panel.setFixedWidth(250)
        self.main_controls_panel.setStyleSheet("""
            background-color: rgba(240, 240, 240, 0.9); 
            border-radius: 8px;
        """)
        controls_layout = QVBoxLayout(self.main_controls_panel)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(15)
        
        self.dropdown_button = QPushButton("Choose Arnis Form")
        self.dropdown_menu = QMenu(self)
        
        # --- START: THE DICTIONARY SOLUTION ---

        # 1. Create a dictionary.
        #    KEY   = The "pretty name" to show the user in the dropdown.
        #    VALUE = The "real name" that your model actually knows.
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

        # 2. Build the dropdown menu from the dictionary's pretty names (the keys).
        for pretty_name in self.practice_stances.keys():
            action = QAction(pretty_name, self)
            # The lambda passes the PRETTY name when a user clicks it
            action.triggered.connect(lambda checked, text=pretty_name: self.on_action_selected(text))
            self.dropdown_menu.addAction(action)

        # --- END OF THE DICTIONARY SOLUTION ---
        
        self.dropdown_button.setMenu(self.dropdown_menu)
        controls_layout.addWidget(self.dropdown_button)
        
        self.user_button = QPushButton("Choose User")
        # ... (rest of user button setup is the same)
        self.user_menu = QMenu(self)
        users = ["Default User", "John Doe", "Jane Smith"]
        for user_text in users:
            action = QAction(user_text, self)
            action.triggered.connect(lambda checked, text=user_text: self.on_user_selected(text))
            self.user_menu.addAction(action)
        self.user_button.setMenu(self.user_menu)
        controls_layout.addWidget(self.user_button)
        
        self.accuracy_label = QLabel("Accuracy: N/A")
        self.remarks_label = QLabel("Remarks: Select a form to begin.")
        controls_layout.addWidget(self.accuracy_label)
        controls_layout.addWidget(self.remarks_label)
        controls_layout.addStretch()
        
        self.view_all_results_button = QPushButton("View All Results")
        self.view_all_results_button.clicked.connect(self.open_results_window)
        controls_layout.addWidget(self.view_all_results_button)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    # --- CHANGE IN THIS FUNCTION ---
    def on_action_selected(self, pretty_name):
        # Set the button to show the pretty name
        self.dropdown_button.setText(pretty_name)
        
        # IMPORTANT: Look up the REAL model name from the dictionary
        # and set it as the target for our comparison logic.
        self.target_form = self.practice_stances[pretty_name]
        
        self.remarks_label.setText(f"Remarks: Now perform {pretty_name}")
        self.remarks_label.setStyleSheet("color: black;")
        self.accuracy_label.setText("Accuracy: N/A")
        print(f"User selected '{pretty_name}', targeting model class: '{self.target_form}'")

    # The rest of your code is perfect and needs no changes
    def resizeEvent(self, event): super().resizeEvent(event); margin=10; panel_height=self.main_controls_panel.sizeHint().height(); self.main_controls_panel.setGeometry(self.width()-self.main_controls_panel.width()-margin, margin, self.main_controls_panel.width(), panel_height)
    def update_frame(self):
        ret, frame = self.cap.read();
        if not ret: return
        frame=cv2.flip(frame, 1); image_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); results=self.pose.process(image_rgb)
        if results.pose_landmarks: self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        if self.target_form and self.model is not None and results.pose_world_landmarks:
            try:
                landmarks=results.pose_world_landmarks.landmark; row=[];
                for lm in landmarks: row.extend([lm.x, lm.y, lm.z])
                X_live=pd.DataFrame([row]); prediction_index=self.model.predict(X_live)[0]; predicted_class=self.class_names[prediction_index]; prediction_proba=self.model.predict_proba(X_live)[0]; confidence=prediction_proba[prediction_index]
                if predicted_class == self.target_form and confidence > 0.60: self.remarks_label.setText("Remarks: Correct!"); self.remarks_label.setStyleSheet("color: green;"); self.accuracy_label.setText(f"Accuracy: {confidence:.2%}")
                else: self.remarks_label.setText(f"Remarks: Adjust for your chosen form"); self.remarks_label.setStyleSheet("color: red;"); self.accuracy_label.setText("Accuracy: N/A")
            except Exception as e: print(f"Error during prediction: {e}"); self.remarks_label.setText("Remarks: Error")
        elif not self.target_form: self.remarks_label.setText("Remarks: Select a form to begin."); self.remarks_label.setStyleSheet("color: black;")
        else: self.remarks_label.setText("Remarks: No pose detected."); self.remarks_label.setStyleSheet("color: black;")
        frame_for_display=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch=frame_for_display.shape; bytes_per_line=ch * w; qt_image=QImage(frame_for_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888); self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    def on_user_selected(self, username): self.user_button.setText(username); print(f"User changed to: {username}"); self.remarks_label.setText("Remarks: Select a form to begin."); self.remarks_label.setStyleSheet("color: black;"); self.accuracy_label.setText("Accuracy: N/A"); self.dropdown_button.setText("Choose Arnis Form"); self.target_form=None
    def open_results_window(self):
        if self.results_win is None or not self.results_win.isVisible(): self.results_win=ResultsWindow(); self.results_win.setFixedSize(1280, 720); self.results_win.show()
    def closeEvent(self, event): self.cap.release(); self.pose.close(); super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()