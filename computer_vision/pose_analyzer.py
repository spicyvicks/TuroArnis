import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from sort import Sort
from duo.duo_instance import ArnisClassifiers

class PoseAnalyzer:
    def __init__(self, detection_interval=3):
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt') # Person detector

        # --- NEW: LOAD THE CUSTOM STICK DETECTOR ---
        try:
            stick_model_path = os.path.join(project_root, 'models', 'stick_detector.pt')
            if not os.path.exists(stick_model_path):
                print(f"[warning] Stick detector model not found at: {stick_model_path}. Stick analysis will be disabled.")
                self.stick_model = None
            else:
                self.stick_model = YOLO(stick_model_path)
                print("[info] custom stick detector loaded successfully.")
        except Exception as e:
            print(f"[critical] could not load stick detector model: {e}")
            self.stick_model = None

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
            model_path = os.path.join(project_root, 'models', 'arnis_classifiers.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Combined model file not found. Expected at: {model_path}")
            self.classifier = joblib.load(model_path)
            print("[info] hybrid model wrapper loaded successfully.")
        except Exception as e:
            print(f"[critical] could not load models: {e}")
            self.classifier = None
        
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.last_detections = []
        print("[info] computer vision components ready.")

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _get_stick_orientation(self, frame, stick_bbox):
        x1, y1, x2, y2 = stick_bbox
        stick_roi = frame[y1:y2, x1:x2]
        if stick_roi.size == 0: return None
        
        hsv_roi = cv2.cvtColor(stick_roi, cv2.COLOR_BGR2HSV)
        # IMPORTANT: Tune these HSV values for your stick's color and lighting!
        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_roi, lower_brown, upper_brown)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50: return None
        
        rect = cv2.minAreaRect(largest_contour)
        box = np.int0(cv2.boxPoints(rect))
        
        side1_len = np.linalg.norm(box[0] - box[1])
        side2_len = np.linalg.norm(box[1] - box[2])
        pt1, pt2 = ((box[1] + box[2]) // 2, (box[0] + box[3]) // 2) if side1_len > side2_len else ((box[0] + box[1]) // 2, (box[2] + box[3]) // 2)
        
        endpoint1 = (pt1[0] + x1, pt1[1] + y1)
        endpoint2 = (pt2[0] + x1, pt2[1] + y1)
        return (endpoint1, endpoint2)

    def _calculate_angle_2d(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc)
        magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
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
        
        stick_boxes = []
        if self.stick_model:
            results_stick = self.stick_model(frame, verbose=False, conf=0.4, imgsz=320)
            for r in results_stick:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    stick_boxes.append((x1, y1, x2, y2))
            
        analysis_results = {
            int(p[4]): {
                'id': int(p[4]), 'bbox': tuple(map(int, p[:4])), 'predicted_class': "N/A",
                'confidence': 0.0, 'live_angles': None, 'landmarks': None,
                'stick_endpoints': None, 'grip_angle': None
            } for p in tracked_persons
        }

        if stick_boxes and tracked_persons is not None:
            for stick_box in stick_boxes:
                best_iou, best_match_id = 0.0, -1
                for person in tracked_persons:
                    person_box = tuple(map(int, person[:4]))
                    iou = self._calculate_iou(stick_box, person_box)
                    if iou > best_iou:
                        best_iou, best_match_id = iou, int(person[4])
                
                if best_match_id != -1 and best_iou > 0.05 and analysis_results[best_match_id]['stick_endpoints'] is None:
                    endpoints = self._get_stick_orientation(frame, stick_box)
                    if endpoints:
                        analysis_results[best_match_id]['stick_endpoints'] = endpoints
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)

        if pose_results.pose_landmarks and tracked_persons is not None and len(tracked_persons) > 0:
            landmarks = pose_results.pose_landmarks.landmark
            min_x, max_x, min_y, max_y = w, 0, h, 0
            for lm in landmarks:
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
                live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks.landmark)
                predicted_class, confidence = "N/A", 0.0

                if self.classifier and live_angles:
                    try:
                        feature_columns = ['left_elbow', 'left_shoulder', 'left_hip', 'left_knee', 'right_elbow', 'right_shoulder', 'right_hip', 'right_knee']
                        ordered_angles = [live_angles[key] for key in feature_columns]
                        live_features_df = pd.DataFrame([ordered_angles], columns=feature_columns)
                        prediction_result = self.classifier.predict(live_features_df.values)
                        predicted_class, confidence = prediction_result['predicted_class'], prediction_result['confidence']
                    except Exception as e:
                        print(f"Error during prediction for user {best_match_id}: {e}")
                
                analysis_results[best_match_id].update({
                    'landmarks': pose_results.pose_landmarks, 'live_angles': live_angles,
                    'predicted_class': predicted_class, 'confidence': confidence
                })

                stick_endpoints = analysis_results[best_match_id]['stick_endpoints']
                if stick_endpoints:
                    r_wrist_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    r_shoulder_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    wrist_pt = (int(r_wrist_lm.x * w), int(r_wrist_lm.y * h))
                    shoulder_pt = (int(r_shoulder_lm.x * w), int(r_shoulder_lm.y * h))
                    
                    dist0 = np.linalg.norm(np.array(wrist_pt) - np.array(stick_endpoints[0]))
                    dist1 = np.linalg.norm(np.array(wrist_pt) - np.array(stick_endpoints[1]))
                    tip = stick_endpoints[0] if dist0 > dist1 else stick_endpoints[1]
                    
                    grip_angle = self._calculate_angle_2d(shoulder_pt, wrist_pt, tip)
                    analysis_results[best_match_id]['grip_angle'] = grip_angle

        return list(analysis_results.values())

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