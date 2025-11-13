import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf

from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import InputLayer

from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from sort import Sort

class CustomInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs['input_shape'] = batch_shape[1:]
        super().__init__(**kwargs)

class PoseAnalyzer:
    def __init__(self, detection_interval=3):
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        self.tracker = Sort(max_age=90, min_hits=3, iou_threshold=0.3)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

        try:
            model_path = os.path.join(project_root, 'models', 'arnis_coordinates_classifier.keras')
            encoder_path = os.path.join(project_root, 'models', 'label_encoder.joblib')

            if not os.path.exists(model_path) or not os.path.exists(encoder_path):
                raise FileNotFoundError("Model or encoder file not found in the 'models' directory.")

            with custom_object_scope({'InputLayer': CustomInputLayer}):
                self.pose_classifier_model = tf.keras.models.load_model(model_path)

            self.label_encoder = joblib.load(encoder_path)
            
            if not self.pose_classifier_model.optimizer:
                self.pose_classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            print("[info] Keras pose classification model and encoder loaded successfully.")
        except Exception as e:
            print(f"[critical] could not load Keras model or encoder: {e}")
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
        return interArea / float(boxAArea + boxBArea - interArea)

    def _detect_stick_by_shape(self, frame, landmarks_2d, frame_shape):

        h, w = frame_shape
        try:
            r_wrist = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            r_elbow = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_shoulder = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            x_coords = [r_wrist.x, r_elbow.x, r_shoulder.x]
            y_coords = [r_wrist.y, r_elbow.y, r_shoulder.y]
            roi_x1 = int(min(x_coords) * w) - 80; roi_y1 = int(min(y_coords) * h) - 80
            roi_x2 = int(max(x_coords) * w) + 80; roi_y2 = int(max(y_coords) * h) + 80
            roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
            roi_x2, roi_y2 = min(w, roi_x2), min(h, roi_y2)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0: return None, None

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            
            # Use Canny edge detection for better stick outline
            edges = cv2.Canny(blurred, 50, 150)
            # Dilate edges to connect broken lines
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_stick_contour = None
            max_length = 0
            candidates = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100: continue  # Reduced from 200
                
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (width, height), angle = rect
                if width > height: width, height = height, width
                
                if width > 0 and height > 0:
                    aspect_ratio = height / width
                    # Reduced aspect ratio from 3 to 2 for better detection
                    if aspect_ratio > 2 and height > max_length:
                        candidates += 1
                        max_length = height
                        best_stick_contour = cnt
            
            if best_stick_contour is not None:
                final_rect = cv2.minAreaRect(best_stick_contour)
                stick_roi_bbox = cv2.boundingRect(best_stick_contour)
                sx, sy, sw, sh = stick_roi_bbox
                stick_frame_bbox = (sx + roi_x1, sy + roi_y1, sw, sh)
                
                box = cv2.boxPoints(final_rect)
                side1_len = np.linalg.norm(box[0] - box[1])
                side2_len = np.linalg.norm(box[1] - box[2])
                pt1, pt2 = ((box[1] + box[2]) // 2, (box[0] + box[3]) // 2) if side1_len > side2_len else ((box[0] + box[1]) // 2, (box[2] + box[3]) // 2)
                
                # Convert to frame coordinates
                endpoint1_frame = (int(pt1[0] + roi_x1), int(pt1[1] + roi_y1))
                endpoint2_frame = (int(pt2[0] + roi_x1), int(pt2[1] + roi_y1))
                
                # Extend the stick line to make it longer (moderate extension)
                dx = endpoint2_frame[0] - endpoint1_frame[0]
                dy = endpoint2_frame[1] - endpoint1_frame[1]
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    # Normalize direction
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    # Extend by 80 pixels from detected portion
                    extension = 80
                    extended_pt1 = (int(endpoint1_frame[0] - dx_norm * extension), 
                                   int(endpoint1_frame[1] - dy_norm * extension))
                    extended_pt2 = (int(endpoint2_frame[0] + dx_norm * extension), 
                                   int(endpoint2_frame[1] + dy_norm * extension))
                    
                    return (extended_pt1, extended_pt2), stick_frame_bbox
                else:
                    return (endpoint1_frame, endpoint2_frame), stick_frame_bbox
            else:
                print(f"[DEBUG] No stick contour found. Checked {len(contours)} contours, {candidates} candidates (aspect > 3)")
        except Exception as e:
            print(f"[ERROR] Stick detection failed: {e}")
        return None, None

    def process_frame(self, frame):
        h, w, _ = frame.shape

        results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0], conf=0.5, imgsz=320)
        detections = np.empty((0, 5))
        for r in results_yolo:
            for box in r.boxes:
                if box.conf[0] >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, box.conf[0]])))
        
        tracked_persons = self.tracker.update(detections)
        
        analysis_results = { 
            int(p[4]): {
                'id': int(p[4]), 'bbox': tuple(map(int, p[:4])), 
                'predicted_class': "N/A", 'confidence': 0.0, 
                'live_angles': None, 'landmarks': None, 
                'stick_endpoints': None, 'stick_keypoints': None, 'grip_angle': None 
            } for p in tracked_persons 
        }
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)

        if pose_results.pose_landmarks and len(tracked_persons) > 0:
            landmarks_2d = pose_results.pose_landmarks.landmark
            min_x, max_x = w, 0; min_y, max_y = h, 0
            for lm in landmarks_2d:
                px, py = int(lm.x * w), int(lm.y * h)
                min_x, max_x = min(min_x, px), max(max_x, px)
                min_y, max_y = min(min_y, py), max(max_y, py)
            mp_box = (min_x, min_y, max_x, max_y)
            
            best_iou, best_match_id = 0.0, -1
            for person_id, data in analysis_results.items():
                iou = self._calculate_iou(mp_box, data['bbox'])
                if iou > best_iou: best_iou, best_match_id = iou, person_id
            
            if best_match_id != -1 and best_iou > 0.3:
                # 5. Detect stick, merge bounding box, and create aligned keypoints
                stick_endpoints, stick_bbox = self._detect_stick_by_shape(frame, landmarks_2d, (h, w))
                if stick_endpoints:
                    analysis_results[best_match_id]['stick_endpoints'] = stick_endpoints
                    user_x1, user_y1, user_x2, user_y2 = analysis_results[best_match_id]['bbox']
                    stick_x, stick_y, stick_w, stick_h = stick_bbox
                    combined_x1 = min(user_x1, stick_x)
                    combined_y1 = min(user_y1, stick_y)
                    combined_x2 = max(user_x2, stick_x + stick_w)
                    combined_y2 = max(user_y2, stick_y + stick_h)
                    analysis_results[best_match_id]['bbox'] = (combined_x1, combined_y1, combined_x2, combined_y2)

                    r_wrist_lm = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    wrist_pt = np.array([int(r_wrist_lm.x * w), int(r_wrist_lm.y * h)])
                    pt1, pt2 = np.array(stick_endpoints[0]), np.array(stick_endpoints[1])
                    grip_pt, tip_pt = (tuple(pt1), tuple(pt2)) if np.linalg.norm(pt1 - wrist_pt) < np.linalg.norm(pt2 - wrist_pt) else (tuple(pt2), tuple(pt1))
                    analysis_results[best_match_id]['stick_keypoints'] = {'grip': grip_pt, 'tip': tip_pt}

                    r_shoulder_lm = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    shoulder_pt = (int(r_shoulder_lm.x * w), int(r_shoulder_lm.y * h))
                    analysis_results[best_match_id]['grip_angle'] = self._calculate_angle_2d(shoulder_pt, wrist_pt, tip_pt)
                
                # 6. Predict pose with Keras model using 3D landmarks
                predicted_class, confidence = "N/A", 0.0
                if self.pose_classifier_model and self.label_encoder:
                    try:
                        world_landmarks = pose_results.pose_world_landmarks.landmark
                        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
                        hip_center = (landmarks_np[23] + landmarks_np[24]) / 2.0
                        coords = (landmarks_np - hip_center).flatten()
                        pred_proba = self.pose_classifier_model.predict(np.expand_dims(coords, axis=0), verbose=0)[0]
                        pred_index = np.argmax(pred_proba)
                        confidence = pred_proba[pred_index]
                        predicted_class = self.label_encoder.inverse_transform([pred_index])[0]
                    except Exception as e:
                        # print(f"Keras prediction failed: {e}")
                        pass
                
                # 7. Calculate all 3D joint angles for detailed feedback
                live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks)
                analysis_results[best_match_id].update({
                    'landmarks': pose_results.pose_landmarks, 
                    'live_angles': live_angles, 
                    'predicted_class': predicted_class, 
                    'confidence': float(confidence)
                })

        return list(analysis_results.values())

    def _calculate_all_angles_3d(self, landmark_list):
        if not landmark_list: return None
        landmarks = landmark_list.landmark
        lm_data = {self.mp_pose.PoseLandmark(i).name.lower(): [landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))}
        try:
            return {
                'left_elbow': self._calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_elbow'], lm_data['left_wrist']),
                'right_elbow': self._calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_elbow'], lm_data['right_wrist']),
                'left_shoulder': self._calculate_angle_3d(lm_data['left_hip'], lm_data['left_shoulder'], lm_data['left_elbow']),
                'right_shoulder': self._calculate_angle_3d(lm_data['right_hip'], lm_data['right_shoulder'], lm_data['right_elbow']),
                'left_knee': self._calculate_angle_3d(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle']),
                'right_knee': self._calculate_angle_3d(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle']),
            }
        except Exception: return None

    def _calculate_angle_3d(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc)
        magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
        return np.degrees(np.arccos(np.clip(dot_product / (magnitude + 1e-6), -1.0, 1.0)))
        
    def _calculate_angle_2d(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc)
        magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
        return np.degrees(np.arccos(np.clip(dot_product / (magnitude + 1e-6), -1.0, 1.0)))

    def close(self):
        self.pose.close()
        print("[info] pose analyzer closed.")