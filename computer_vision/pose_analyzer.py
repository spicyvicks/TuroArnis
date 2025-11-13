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

    def _detect_stick_by_shape(self, frame, landmarks_2d, frame_shape, person_bbox=None):

        h, w = frame_shape
        try:
            r_wrist = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            r_elbow = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_shoulder = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_knee = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            
            # Create a large ROI around the entire right side to capture stick in any orientation
            wrist_x, wrist_y = int(r_wrist.x * w), int(r_wrist.y * h)
            elbow_x, elbow_y = int(r_elbow.x * w), int(r_elbow.y * h)
            shoulder_x, shoulder_y = int(r_shoulder.x * w), int(r_shoulder.y * h)
            knee_x, knee_y = int(r_knee.x * w), int(r_knee.y * h)
            
            # Get bounding box of right arm + generous padding for stick
            x_coords = [wrist_x, elbow_x, shoulder_x, knee_x]
            y_coords = [wrist_y, elbow_y, shoulder_y, knee_y]
            
            # Large padding to ensure stick is captured regardless of pose orientation
            padding = 200
            roi_x1 = max(0, min(x_coords) - padding)
            roi_y1 = max(0, min(y_coords) - padding)
            roi_x2 = min(w, max(x_coords) + padding)
            roi_y2 = min(h, max(y_coords) + padding)
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
                
                # Get angle from detected contour
                (cx, cy), (w_rect, h_rect), angle = final_rect
                angle_rad = np.deg2rad(angle)
                if w_rect > h_rect:
                    angle_rad += np.pi / 2
                
                # Use wrist as anchor and calculate stick length from body proportions
                wrist_x, wrist_y = int(r_wrist.x * w), int(r_wrist.y * h)
                shoulder_x, shoulder_y = int(r_shoulder.x * w), int(r_shoulder.y * h)
                knee_x, knee_y = int(r_knee.x * w), int(r_knee.y * h)
                shoulder_knee_dist = np.sqrt((shoulder_x - knee_x)**2 + (shoulder_y - knee_y)**2)
                
                # Calculate elbow-to-wrist direction
                elbow_x, elbow_y = int(r_elbow.x * w), int(r_elbow.y * h)
                arm_dx = wrist_x - elbow_x
                arm_dy = wrist_y - elbow_y
                arm_angle = np.arctan2(arm_dy, arm_dx)
                
                # Calculate stick direction from detected angle
                stick_dx = np.cos(angle_rad)
                stick_dy = np.sin(angle_rad)
                
                # Determine which direction the stick extends
                angle_diff1 = abs(angle_rad - arm_angle)
                angle_diff2 = abs((angle_rad + np.pi) - arm_angle)
                
                # Normalize to [0, pi]
                angle_diff1 = min(angle_diff1, 2*np.pi - angle_diff1)
                angle_diff2 = min(angle_diff2, 2*np.pi - angle_diff2)
                
                # Flip if opposite direction aligns better with arm
                if angle_diff2 < angle_diff1:
                    stick_dx = -stick_dx
                    stick_dy = -stick_dy
                
                # Draw stick with body-proportional length
                stick_length = int(shoulder_knee_dist)
                endpoint1 = (int(wrist_x - stick_dx * 30),
                           int(wrist_y - stick_dy * 30))
                endpoint2 = (int(wrist_x + stick_dx * stick_length),
                           int(wrist_y + stick_dy * stick_length))
                
                return (endpoint1, endpoint2), stick_frame_bbox
            else:
                # Fallback: No stick detected - use arm direction as estimate
                wrist_x, wrist_y = int(r_wrist.x * w), int(r_wrist.y * h)
                elbow_x, elbow_y = int(r_elbow.x * w), int(r_elbow.y * h)
                shoulder_x, shoulder_y = int(r_shoulder.x * w), int(r_shoulder.y * h)
                knee_x, knee_y = int(r_knee.x * w), int(r_knee.y * h)
                
                # Calculate arm direction
                arm_dx = wrist_x - elbow_x
                arm_dy = wrist_y - elbow_y
                arm_length = np.sqrt(arm_dx*arm_dx + arm_dy*arm_dy)
                
                if arm_length > 0:
                    arm_dx_norm = arm_dx / arm_length
                    arm_dy_norm = arm_dy / arm_length
                    
                    # Stick extends in arm direction
                    shoulder_knee_dist = np.sqrt((shoulder_x - knee_x)**2 + (shoulder_y - knee_y)**2)
                    stick_length = int(shoulder_knee_dist)
                    
                    endpoint1 = (int(wrist_x - arm_dx_norm * 30),
                               int(wrist_y - arm_dy_norm * 30))
                    endpoint2 = (int(wrist_x + arm_dx_norm * stick_length),
                               int(wrist_y + arm_dy_norm * stick_length))
                    
                    return (endpoint1, endpoint2), None
                
                return None, None
                
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
                person_bbox = analysis_results[best_match_id]['bbox']
                stick_endpoints, stick_bbox = self._detect_stick_by_shape(frame, landmarks_2d, (h, w), person_bbox)
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