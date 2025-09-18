# file: computer_vision/pose_analyzer.py
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from ultralytics import YOLO
from sort import Sort

class PoseAnalyzer:
    def __init__(self):
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt')
        # Modify SORT parameters:
        # max_age: how many frames a track can be missing before being deleted
        # min_hits: minimum number of detections before track is initialized
        # iou_threshold: intersection over union threshold for matching
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # load the arnis classifier model
        try:
            self.classifier_model = joblib.load('models/arnis_random_forest_classifier.joblib')
            self.class_names = joblib.load('models/arnis_class_names.joblib')
            print("[info] classifier model loaded.")
        except FileNotFoundError:
            self.classifier_model = None
            self.class_names = None
            print("[error] classifier model not found. analysis will be limited.")
        print("[info] computer vision components ready.")

    def process_frame(self, frame, target_form=None):
        # 1. detect and track persons with yolo and sort
        results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0])
        detections = np.empty((0, 5))

        # Set minimum confidence threshold
        min_confidence = 0.5

        for r in results_yolo:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Only track high confidence detections
                if conf >= min_confidence:
                    # Ensure coordinates are within frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Only add valid detections
                    if x2 > x1 and y2 > y1:
                        detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

        # Update tracker with filtered detections
        tracked_persons = self.tracker.update(detections)

        # 2. process each tracked person
        for person in tracked_persons:
            x1, y1, x2, y2, person_id = map(int, person)
            
            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Check if we have a valid region to crop
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
                
            # extract the person's region
            try:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                    
                # process with mediapipe pose
                image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(image_rgb)
                
                # draw bounding box and id for everyone
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f"User {person_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                
                # draw pose landmarks if detected
                if pose_results.pose_landmarks:
                    self._draw_landmarks_on_main_frame(frame, pose_results, x1, y1, x2, y2)
                    
                    # run analysis only if a form is selected
                    if target_form and self.classifier_model:
                        landmarks = pose_results.pose_world_landmarks.landmark
                        row = []
                        for lm in landmarks:
                            row.extend([lm.x, lm.y, lm.z])
                        
                        # Make prediction
                        X_live = pd.DataFrame([row])
                        prediction_index = self.classifier_model.predict(X_live)[0]
                        predicted_class = self.class_names[prediction_index]
                        
                        # Draw prediction results
                        prediction_text = f"Form: {predicted_class}"
                        cv2.putText(frame, prediction_text, (x1, y2 + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing person {person_id}: {e}")
                continue

        return frame

    def _draw_landmarks_on_main_frame(self, main_frame, pose_results, x1, y1, x2, y2):
        # this is a "private" helper method for drawing
        # Create a new NormalizedLandmarkList using the pose module
        translated_landmarks = self.mp_pose.PoseLandmark._member_names_
        landmarks_proto = pose_results.pose_landmarks
        
        frame_height, frame_width, _ = main_frame.shape
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Translate the landmarks to match the main frame coordinates
        for landmark in landmarks_proto.landmark:
            pixel_x = landmark.x * crop_width + x1
            pixel_y = landmark.y * crop_height + y1
            
            landmark.x = pixel_x / frame_width
            landmark.y = pixel_y / frame_height

        # Draw the landmarks on the main frame
        self.mp_drawing.draw_landmarks(
            main_frame,
            landmarks_proto,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    def close(self):
        # method to clean up resources
        self.pose.close()
        print("[info] pose analyzer closed.")