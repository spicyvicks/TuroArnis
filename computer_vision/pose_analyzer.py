import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import tensorflow as tf

# Keras compatibility imports for the fix
from tensorflow.keras.utils import custom_object_scope # type: ignore [attr-defined]
from tensorflow.keras.layers import InputLayer # type: ignore

# Add suppression for Ultralytics
from ultralytics import YOLO # type: ignore 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from sort import Sort

# Custom class to handle older Keras model format
class CustomInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs['input_shape'] = batch_shape[1:] 
        super().__init__(**kwargs)

class PoseAnalyzer:
    def __init__(self, detection_interval=3, **kwargs): # Removed enable_stick_detection
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt') 
        
        # --- STICK MODEL REMOVED ---
        print("[info] Using heuristic-based stick detection (no model needed).")

        self.tracker = Sort(max_age=90, min_hits=3, iou_threshold=0.3)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        try:
            model_path = os.path.join(project_root, 'models', 'arnis_coordinates_classifier.keras')
            encoder_path = os.path.join(project_root, 'models', 'label_encoder.joblib')

            if not os.path.exists(model_path) or not os.path.exists(encoder_path):
                raise FileNotFoundError("Model or encoder file not found.")

            with custom_object_scope({'InputLayer': CustomInputLayer}):
                self.pose_classifier_model = tf.keras.models.load_model(model_path)

            self.label_encoder = joblib.load(encoder_path)
            
            if not self.pose_classifier_model.optimizer:
                self.pose_classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            print("[info] Keras pose classification model and encoder loaded successfully.")
        except Exception as e:
            print(f"[critical] could not load new models: {e}")
            self.pose_classifier_model = None
            self.label_encoder = None
        
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.last_detections = []
        print("[info] computer vision components ready.")

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    # --- NEW HEURISTIC STICK DETECTOR ---
    def _detect_stick_heuristically(self, frame, landmarks_2d, frame_shape):
        h, w = frame_shape
        try:
            # Anchor points: right wrist and right elbow
            r_wrist = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            r_elbow = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            
            # Define a Region of Interest (ROI) around the hand/forearm
            x1 = int(min(r_wrist.x, r_elbow.x) * w) - 50
            x2 = int(max(r_wrist.x, r_elbow.x) * w) + 50
            y1 = int(min(r_wrist.y, r_elbow.y) * h) - 50
            y2 = int(max(r_wrist.y, r_elbow.y) * h) + 50

            # Clamp coordinates to be within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: return None

            # Convert ROI to HSV for color segmentation
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define HSV color range for brown/tan (this may need tuning)
            lower_brown = np.array([5, 40, 40])
            upper_brown = np.array([30, 255, 255])
            
            # Create a mask to isolate the stick color
            mask = cv2.inRange(hsv_roi, lower_brown, upper_brown)
            
            # Find contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None

            best_contour = None
            max_aspect_ratio = 4 # A stick should be at least 4 times longer than it is wide

            for cnt in contours:
                # Filter out small noise
                if cv2.contourArea(cnt) < 100: continue
                
                # Get the minimum area rectangle around the contour
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (width, height), angle = rect
                
                # Ensure width is the smaller dimension
                if width > height:
                    width, height = height, width
                
                if width > 0:
                    aspect_ratio = height / width
                    if aspect_ratio > max_aspect_ratio:
                        max_aspect_ratio = aspect_ratio
                        best_contour = cnt
            
            if best_contour is not None:
                # Recalculate the final rectangle for the best contour
                final_rect = cv2.minAreaRect(best_contour)
                box = cv2.boxPoints(final_rect)
                box = np.int0(box)

                # Find the two endpoints of the stick's longest axis
                side1_len = np.linalg.norm(box[0] - box[1])
                side2_len = np.linalg.norm(box[1] - box[2])
                
                if side1_len > side2_len:
                    pt1 = (box[1] + box[2]) // 2
                    pt2 = (box[0] + box[3]) // 2
                else:
                    pt1 = (box[0] + box[1]) // 2
                    pt2 = (box[2] + box[3]) // 2
                
                # Translate endpoints from ROI coordinates back to full frame coordinates
                return (pt1[0] + x1, pt1[1] + y1), (pt2[0] + x1, pt2[1] + y1)

        except Exception:
            return None
        return None

    def _calculate_angle_2d(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc); magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
        if magnitude < 1e-6: return 0.0
        cosine_angle = np.clip(dot_product / magnitude, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))
    
    def process_frame(self, frame):
        self.frame_count += 1
        h, w, _ = frame.shape

        if self.frame_count % self.detection_interval == 0:
            results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0], conf=0.5, imgsz=320)
            detections = np.empty((0, 5))
            for r in results_yolo:
                for box in r.boxes:
                    if box.conf[0] >= 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections = np.vstack((detections, np.array([x1, y1, x2, y2, box.conf[0]])))
            tracked_persons = self.tracker.update(detections)
            self.last_detections = tracked_persons
        else:
            tracked_persons = self.last_detections if self.last_detections is not None else []
        
        analysis_results = { int(p[4]): {'id': int(p[4]), 'bbox': tuple(map(int, p[:4])), 'predicted_class': "N/A", 'confidence': 0.0, 'live_angles': None, 'landmarks': None, 'stick_endpoints': None, 'grip_angle': None } for p in tracked_persons }
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)

        if pose_results.pose_landmarks and tracked_persons is not None and len(tracked_persons) > 0:
            landmarks_2d = pose_results.pose_landmarks.landmark
            min_x, max_x, min_y, max_y = w, 0, h, 0
            for lm in landmarks_2d:
                px, py = int(lm.x * w), int(lm.y * h)
                min_x, max_x = min(min_x, px), max(max_x, px)
                min_y, max_y = min(min_y, py), max(max_y, py)
            mp_box = (min_x, min_y, max_x, max_y)
            
            best_iou, best_match_id = 0.0, -1
            for person_id, data in analysis_results.items():
                iou = self._calculate_iou(mp_box, data['bbox'])
                if iou > best_iou:
                    best_iou, best_match_id = iou, person_id
            
            if best_match_id != -1 and best_iou > 0.3:
                
                # --- HEURISTIC STICK DETECTION CALL ---
                stick_endpoints = self._detect_stick_heuristically(frame, landmarks_2d, (h, w))
                if stick_endpoints:
                    analysis_results[best_match_id]['stick_endpoints'] = stick_endpoints

                predicted_class, confidence = "N/A", 0.0
                if self.pose_classifier_model and self.label_encoder:
                    try:
                        if not pose_results.pose_world_landmarks:
                            raise ValueError("MediaPipe did not return 3D world landmarks.")
                        world_landmarks = pose_results.pose_world_landmarks.landmark
                        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
                        
                        if len(landmarks_np) != 33:
                            raise ValueError(f"MediaPipe returned {len(landmarks_np)} landmarks, expected 33.")

                        left_hip_idx = 23; right_hip_idx = 24
                        hip_center = (landmarks_np[left_hip_idx] + landmarks_np[right_hip_idx]) / 2.0
                        normalized_landmarks = landmarks_np - hip_center
                        coords = normalized_landmarks.flatten()
                        
                        if coords.shape[0] != 99:
                            raise ValueError(f"Feature vector shape is {coords.shape[0]}, expected 99.")
                        
                        coords_batch = np.expand_dims(coords, axis=0)
                        pred_proba = self.pose_classifier_model.predict(coords_batch, verbose=0)[0]
                        pred_index = np.argmax(pred_proba)
                        confidence = pred_proba[pred_index]
                        predicted_class = self.label_encoder.inverse_transform([pred_index])[0]
                        
                    except Exception as e:
                        print(f"[ERROR] Prediction failed for user {best_match_id}: {e}")
                
                live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks)
                analysis_results[best_match_id].update({
                    'landmarks': pose_results.pose_landmarks, 
                    'live_angles': live_angles, 
                    'predicted_class': predicted_class, 
                    'confidence': float(confidence)
                })

                if stick_endpoints:
                    r_wrist_lm = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    r_shoulder_lm = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    wrist_pt = (int(r_wrist_lm.x * w), int(r_wrist_lm.y * h))
                    shoulder_pt = (int(r_shoulder_lm.x * w), int(r_shoulder_lm.y * h))
                    dist0 = np.linalg.norm(np.array(wrist_pt) - np.array(stick_endpoints[0]))
                    dist1 = np.linalg.norm(np.array(wrist_pt) - np.array(stick_endpoints[1]))
                    tip = stick_endpoints[0] if dist0 > dist1 else stick_endpoints[1]
                    grip_angle = self._calculate_angle_2d(shoulder_pt, wrist_pt, tip)
                    analysis_results[best_match_id]['grip_angle'] = grip_angle

        return list(analysis_results.values())

    def _calculate_all_angles_3d(self, landmark_list):
        if not landmark_list: return None
        landmarks = landmark_list.landmark
        try:
            lm_data = {self.mp_pose.PoseLandmark(i).name.lower(): [landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))}
            return {
                'left_elbow': self._calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_elbow'], lm_data['left_wrist']),
                'right_elbow': self._calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_elbow'], lm_data['right_wrist']),
                'left_shoulder': self._calculate_angle_3d(lm_data['left_hip'], lm_data['left_shoulder'], lm_data['left_elbow']),
                'right_shoulder': self._calculate_angle_3d(lm_data['right_hip'], lm_data['right_shoulder'], lm_data['right_elbow']),
                # ... add other angles as needed ...
            }
        except Exception:
            return None

    def _calculate_angle_3d(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc)
        magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
        cosine_angle = np.clip(dot_product / (magnitude + 1e-6), -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def close(self):
        self.pose.close()
        print("[info] pose analyzer closed.")