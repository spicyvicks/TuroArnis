import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf

from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from sort import Sort

class PoseAnalyzer:
    def __init__(self, detection_interval=3, stick_model_path=None, debug_stick=False):
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        self.stick_detector = None
        self.debug_stick = debug_stick  
        print(f"[DEBUG-INIT] Stick model path provided: {stick_model_path}")
        print(f"[DEBUG-INIT] Path exists: {os.path.exists(stick_model_path) if stick_model_path else 'N/A'}")
        print(f"[DEBUG-INIT] Debug stick enabled: {debug_stick}")
        
        if stick_model_path and os.path.exists(stick_model_path):
            try:
                self.stick_detector = YOLO(stick_model_path)
                print(f"[info] Stick detector model loaded from {stick_model_path}")
                print(f"[DEBUG-INIT] Stick detector type: {type(self.stick_detector)}")
                print(f"[DEBUG-INIT] Stick detector model names: {self.stick_detector.names if hasattr(self.stick_detector, 'names') else 'N/A'}")
            except Exception as e:
                print(f"[warning] Could not load stick detector: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG-INIT] Stick detector NOT loaded - path is None or doesn't exist")
        
        self.tracker = Sort(
            max_age=120,
            min_hits=2,
            iou_threshold=0.25
        )
        self.id_mapping = {}
        self.next_stable_id = 1
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        # PERFORMANCE OPTIMIZATION: Use faster settings for video
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Changed from True - much faster for video
            model_complexity=1,        # Changed from 2 - balanced speed/accuracy
            min_detection_confidence=0.5,
            smooth_landmarks=True      # Smoother output for video
        )

        try:
            model_path = os.path.join(project_root, 'models', 'arnis_coordinates_classifier.keras')
            encoder_path = os.path.join(project_root, 'models', 'label_encoder.joblib')

            if not os.path.exists(model_path) or not os.path.exists(encoder_path):
                raise FileNotFoundError("Model or encoder file not found in the 'models' directory.")

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

    def _detect_stick_with_yolo(self, frame, person_bbox=None, debug=False):
        if debug:
            print(f"[DEBUG-STICK] _detect_stick_with_yolo called")
            print(f"[DEBUG-STICK] Frame shape: {frame.shape}")
            print(f"[DEBUG-STICK] Person bbox: {person_bbox}")
            print(f"[DEBUG-STICK] Stick detector is None: {self.stick_detector is None}")
        
        if self.stick_detector is None:
            if debug:
                print("[DEBUG-STICK] EXITING: Stick detector not loaded")
            return None, None
        
        try:
            if debug:
                print(f"[DEBUG-STICK] Running stick detector on frame...")
            
            # Run stick detection
            results = self.stick_detector(frame, verbose=False, conf=0.5)
            
            if debug:
                print(f"[DEBUG-STICK] Results returned: {len(results)} detections")
                print(f"[DEBUG-STICK] Results type: {type(results)}")
            
            if len(results) == 0:
                if debug:
                    print("[DEBUG-STICK] EXITING: No results from detector")
                return None, None
            
            if results[0].keypoints is None:
                if debug:
                    print("[DEBUG-STICK] EXITING: Results[0] has no keypoints")
                    print(f"[DEBUG-STICK] Results[0] boxes count: {len(results[0].boxes) if results[0].boxes is not None else 'None'}")
                return None, None
            
            result = results[0]
            
            if debug:
                print(f"[DEBUG-STICK] First result obtained")
                print(f"[DEBUG-STICK] Boxes: {len(result.boxes) if result.boxes is not None else 'None'}")
            
            if len(result.boxes) == 0:
                if debug:
                    print("[DEBUG-STICK] EXITING: No bounding boxes found")
                return None, None
            
            # Get stick bounding box
            stick_box = result.boxes[0]
            stick_bbox = tuple(map(int, stick_box.xyxy[0].tolist()))
            confidence = stick_box.conf.item()
            
            if debug:
                print(f"[DEBUG-STICK] ✓ Stick detected - Confidence: {confidence:.3f}")
                print(f"[DEBUG-STICK] Stick bbox: {stick_bbox}")
                print(f"[DEBUG-STICK] Stick box class: {stick_box.cls.item() if stick_box.cls is not None else 'None'}")
            
            # Get stick keypoints (grip and tip)
            if result.keypoints is not None and len(result.keypoints) > 0:
                if debug:
                    print(f"[DEBUG-STICK] Keypoints object exists, length: {len(result.keypoints)}")
                    print(f"[DEBUG-STICK] Keypoints type: {type(result.keypoints)}")
                
                kpts = result.keypoints[0].data[0]  # First detection's keypoints
                
                if debug:
                    print(f"[DEBUG-STICK] Keypoints data shape: {kpts.shape if hasattr(kpts, 'shape') else 'N/A'}")
                    print(f"[DEBUG-STICK] Keypoints data: {kpts}")
                
                grip_point = (int(kpts[0][0]), int(kpts[0][1]))
                tip_point = (int(kpts[1][0]), int(kpts[1][1]))
                grip_conf = kpts[0][2].item()
                tip_conf = kpts[1][2].item()
                
                if debug:
                    print(f"[DEBUG-STICK] ✓ Grip keypoint: {grip_point} (conf: {grip_conf:.3f})")
                    print(f"[DEBUG-STICK] ✓ Tip keypoint: {tip_point} (conf: {tip_conf:.3f})")
                    stick_length = np.sqrt((tip_point[0] - grip_point[0])**2 + 
                                         (tip_point[1] - grip_point[1])**2)
                    print(f"[DEBUG-STICK] ✓ Stick length: {stick_length:.1f} pixels")
                    print(f"[DEBUG-STICK] ✓ RETURNING stick_endpoints and bbox")
                
                return (grip_point, tip_point), stick_bbox
            else:
                if debug:
                    print(f"[DEBUG-STICK] EXITING: Keypoints is None or empty")
            
            return None, None
            
        except Exception as e:
            print(f"[ERROR-STICK] YOLO stick detection failed: {e}")
            if debug:
                print(f"[DEBUG-STICK] Exception type: {type(e)}")
                print(f"[DEBUG-STICK] Exception args: {e.args}")
                import traceback
                traceback.print_exc()
            return None, None

    def process_frame(self, frame):
        h, w, _ = frame.shape

        results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0], conf=0.3, imgsz=320)
        detections = np.empty((0, 5))
        for r in results_yolo:
            for box in r.boxes:
                if box.conf[0] >= 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, box.conf[0]])))
        
        tracked_persons = self.tracker.update(detections)
        
        stable_tracked = []
        for person in tracked_persons:
            tracker_id = int(person[4])
            if tracker_id not in self.id_mapping:
                self.id_mapping[tracker_id] = self.next_stable_id
                self.next_stable_id += 1
            stable_id = self.id_mapping[tracker_id]
            stable_person = np.array([person[0], person[1], person[2], person[3], stable_id])
            stable_tracked.append(stable_person)
        
        tracked_persons = np.array(stable_tracked) if stable_tracked else np.empty((0, 5))
        
        analysis_results = { 
            int(p[4]): {
                'id': int(p[4]), 'bbox': tuple(map(int, p[:4])), 
                'predicted_class': "N/A", 'confidence': 0.0, 
                'live_angles': None, 'landmarks': None, 
                'stick_endpoints': None, 'stick_keypoints': None, 'grip_angle': None 
            } for p in tracked_persons 
        }
        
        for person in tracked_persons:
            person_id = int(person[4])
            x1, y1, x2, y2 = map(int, person[:4])
            
            x1_pad = max(0, x1 - 20)
            y1_pad = max(0, y1 - 20)
            x2_pad = min(w, x2 + 20)
            y2_pad = min(h, y2 + 20)
            
            person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if person_crop.size == 0:
                continue
            
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(crop_rgb)
            
            if pose_results.pose_landmarks:
                landmarks_2d = pose_results.pose_landmarks.landmark
                
                offset_x = x1_pad
                offset_y = y1_pad
                crop_h, crop_w = person_crop.shape[:2]
                
                abs_landmarks = []
                for lm in landmarks_2d:
                    abs_x = int(lm.x * crop_w) + offset_x
                    abs_y = int(lm.y * crop_h) + offset_y
                    abs_landmarks.append((abs_x, abs_y, lm.z))
                
                live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks)
                
                predicted_class, confidence = "N/A", 0.0
                if self.pose_classifier_model and self.label_encoder and live_angles:
                    try:
                        world_landmarks = pose_results.pose_world_landmarks.landmark
                        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
                        hip_center = (landmarks_np[23] + landmarks_np[24]) / 2.0
                        coords = (landmarks_np - hip_center).flatten()
                        pred_proba = self.pose_classifier_model.predict(np.expand_dims(coords, axis=0), verbose=0)[0]
                        pred_index = np.argmax(pred_proba)
                        confidence = pred_proba[pred_index]
                        predicted_class = self.label_encoder.inverse_transform([pred_index])[0]
                    except Exception:
                        pass
                
                if self.debug_stick:
                    print(f"[DEBUG-PROCESS] Calling stick detection for person {person_id}")
                    print(f"[DEBUG-PROCESS] Person bbox: {(x1, y1, x2, y2)}")
                
                stick_endpoints, stick_bbox = self._detect_stick_with_yolo(frame, (x1, y1, x2, y2), debug=self.debug_stick)
                
                if self.debug_stick:
                    print(f"[DEBUG-PROCESS] Stick detection returned:")
                    print(f"[DEBUG-PROCESS]   stick_endpoints: {stick_endpoints}")
                    print(f"[DEBUG-PROCESS]   stick_bbox: {stick_bbox}")
                
                if stick_endpoints:
                    if self.debug_stick:
                        print(f"[DEBUG-PROCESS] ✓ Setting stick_endpoints for person {person_id}")
                    analysis_results[person_id]['stick_endpoints'] = stick_endpoints
                    
                    grip_pt, tip_pt = stick_endpoints
                    analysis_results[person_id]['stick_keypoints'] = {'grip': grip_pt, 'tip': tip_pt}

                    r_wrist_lm = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    l_wrist_lm = landmarks_2d[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    r_wrist_pt = np.array([int(r_wrist_lm.x * crop_w) + offset_x, int(r_wrist_lm.y * crop_h) + offset_y])
                    l_wrist_pt = np.array([int(l_wrist_lm.x * crop_w) + offset_x, int(l_wrist_lm.y * crop_h) + offset_y])
                    
                    grip_array = np.array(grip_pt)
                    r_dist = np.linalg.norm(grip_array - r_wrist_pt)
                    l_dist = np.linalg.norm(grip_array - l_wrist_pt)
                    
                    if r_dist < l_dist:
                        shoulder_lm = landmarks_2d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        wrist_pt = r_wrist_pt
                    else:
                        shoulder_lm = landmarks_2d[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                        wrist_pt = l_wrist_pt
                    
                    shoulder_pt = (int(shoulder_lm.x * w), int(shoulder_lm.y * h))
                    stick_vec = np.array(tip_pt) - np.array(grip_pt)
                    arm_vec = np.array(wrist_pt) - np.array(shoulder_pt)
                    
                    dot = np.dot(stick_vec, arm_vec)
                    norm_stick = np.linalg.norm(stick_vec)
                    norm_arm = np.linalg.norm(arm_vec)
                    
                    if norm_stick > 0 and norm_arm > 0:
                        cos_angle = dot / (norm_stick * norm_arm)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        analysis_results[person_id]['grip_angle'] = angle_deg
                else:
                    if self.debug_stick:
                        print(f"[DEBUG-PROCESS] ✗ No stick_endpoints detected")
                
                analysis_results[person_id]['predicted_class'] = predicted_class
                analysis_results[person_id]['confidence'] = confidence
                analysis_results[person_id]['live_angles'] = live_angles
                analysis_results[person_id]['landmarks'] = pose_results.pose_landmarks
                analysis_results[person_id]['landmarks_absolute'] = abs_landmarks

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

    def reset_tracker(self):
        self.tracker = Sort(
            max_age=120,
            min_hits=2,
            iou_threshold=0.25
        )
        self.id_mapping.clear()
        self.next_stable_id = 1
        print("[info] Tracker reset - IDs reinitialized")

    def close(self):
        self.pose.close()
        print("[info] pose analyzer closed.")