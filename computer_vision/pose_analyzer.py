# file: computer_vision/pose_analyzer.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from ultralytics import YOLO
from sort import Sort
import tensorflow as tf
from mediapipe.framework.formats import landmark_pb2

class PoseAnalyzer:
    def __init__(self, **kwargs): 
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt')
        self.tracker = Sort(max_age=90, min_hits=3, iou_threshold=0.3)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        try:
            self.classifier_model = tf.keras.models.load_model('models_tf/arnis_tf_classifier.h5')
            self.label_encoder = joblib.load('models_tf/label_encoder.joblib')
            print("[info] tensorflow model and encoder loaded successfully.")
        except Exception as e:
            print(f"[warning] could not load tensorflow model or encoder: {e}")
            self.classifier_model = None
            self.label_encoder = None
        
        print("[info] computer vision components ready.")

    def process_frame(self, frame):
        analysis_results = []
        
        #detect & track users
        results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0])
        detections = np.empty((0, 5))
        for r in results_yolo:
            for box in r.boxes:
                if box.conf[0] >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, box.conf[0]])))
        
        tracked_persons = self.tracker.update(detections)

        #for each tracked person, analyze pose
        for person in tracked_persons:
            x1, y1, x2, y2, person_id = map(int, person)
            
            #results for each user
            person_result = {
                'id': person_id, 
                'bbox': (x1, y1, x2, y2),
                'predicted_class': None, 
                'confidence': 0.0, 
                'live_angles': None,
                'landmarks': None 
            }
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                analysis_results.append(person_result)
                continue

            image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(image_rgb)
            
            if pose_results.pose_landmarks:
                #store
                person_result['landmarks'] = self._translate_landmarks(pose_results.pose_landmarks, frame, x1, y1, x2, y2)
            
            if self.classifier_model and self.label_encoder and pose_results.pose_world_landmarks:
                try:
                    live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks.landmark)
                    person_result['live_angles'] = live_angles

                    if live_angles:
                        live_features = pd.DataFrame([list(live_angles.values())])
                        prediction_proba = self.classifier_model.predict(live_features, verbose=0)[0]
                        prediction_index = np.argmax(prediction_proba)
                        
                        person_result['confidence'] = prediction_proba[prediction_index]
                        person_result['predicted_class'] = self.label_encoder.inverse_transform([prediction_index])[0]
                
                except Exception as e:
                    print(f"error during tensorflow analysis for user {person_id}: {e}")

            analysis_results.append(person_result)
            
        return analysis_results

    def _translate_landmarks(self, landmarks, main_frame, x1, y1, x2, y2):
        #helper to translate cropped landmarks to main frame coordinates
        frame_height, frame_width, _ = main_frame.shape
        crop_width, crop_height = x2 - x1, y2 - y1
        
        landmarks_copy = landmark_pb2.NormalizedLandmarkList()
        landmarks_copy.CopyFrom(landmarks)

        for landmark in landmarks_copy.landmark:
            pixel_x = landmark.x * crop_width + x1
            pixel_y = landmark.y * crop_height + y1
            landmark.x, landmark.y = pixel_x / frame_width, pixel_y / frame_height
        
        return landmarks_copy

    def _calculate_angle_3d(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc)
        magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
        cosine_angle = np.clip(dot_product / (magnitude + 1e-6), -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def _calculate_all_angles_3d(self, landmarks):
        try:
            lm_data = {self.mp_pose.PoseLandmark(i).name.lower(): [landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))}
            return {
                'left_elbow': self._calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_elbow'], lm_data['left_wrist']),
                'left_shoulder': self._calculate_angle_3d(lm_data['left_hip'], lm_data['left_shoulder'], lm_data['left_elbow']),
                'left_hip': self._calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_hip'], lm_data['left_knee']),
                'left_knee': self._calculate_angle_3d(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle']),
                'right_elbow': self._calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_elbow'], lm_data['right_wrist']),
                'right_shoulder': self._calculate_angle_3d(lm_data['right_hip'], lm_data['right_shoulder'], lm_data['right_elbow']),
                'right_hip': self._calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_hip'], lm_data['right_knee']),
                'right_knee': self._calculate_angle_3d(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle']),
            }
        except Exception:
            return None

    def close(self):
        self.pose.close()
        print("[info] pose analyzer closed.")