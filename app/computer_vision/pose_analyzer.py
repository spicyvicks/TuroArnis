import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp
# TensorFlow and Keras will be lazy-loaded in legacy fallback

from ultralytics import YOLO

#import resource path helper for pyinstaller compatibility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.resource_path import get_resource_path
from utils.device_manager import configure_device, get_yolo_device

# GCN integration imports
from computer_vision.gcn_inference import get_gcn_engine
from models.gcn.feature_extraction import compute_global_features_from_kpts
from models.gcn.model_architecture import CLASS_NAMES

class PoseAnalyzer:
    def __init__(self, detection_interval=3, stick_model_path=None, debug_stick=False):
        print("[info] initializing computer vision components...")
        
        #configure device (gpu/cpu) for tensorflow and pytorch/yolo
        self.device_info = configure_device(verbose=True)
        self.yolo_device = get_yolo_device(self.device_info)
        
        #use resource path helper for pyinstaller compatibility
        yolo_base_path = get_resource_path('yolov8n.pt')
        self.yolo_model = YOLO(yolo_base_path)
        self.yolo_model.to(self.yolo_device)  #move model to gpu if available
        
        #cached stick detection results
        self._cached_stick_results = {}
        
        self.stick_detector = None
        self.debug_stick = debug_stick  
        print(f"[DEBUG-INIT] Stick model path provided: {stick_model_path}")
        print(f"[DEBUG-INIT] Path exists: {os.path.exists(stick_model_path) if stick_model_path else 'N/A'}")
        print(f"[DEBUG-INIT] Debug stick enabled: {debug_stick}")
        
        if stick_model_path and os.path.exists(stick_model_path):
            try:
                self.stick_detector = YOLO(stick_model_path)
                self.stick_detector.to(self.yolo_device)  #move model to gpu if available
                print(f"[info] Stick detector model loaded from {stick_model_path}")
                print(f"[DEBUG-INIT] Stick detector type: {type(self.stick_detector)}")
                print(f"[DEBUG-INIT] Stick detector model names: {self.stick_detector.names if hasattr(self.stick_detector, 'names') else 'N/A'}")
            except Exception as e:
                print(f"[warning] Could not load stick detector: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG-INIT] Stick detector NOT loaded - path is None or doesn't exist")
        
        #using ultralytics bytetrack
        self.use_builtin_tracking = True
        self.track_history = {}
        self.id_mapping = {}
        self.next_stable_id = 1
        print("[info] using bytetrack for person tracking")
        
        #stick keypoint smoothing buffer
        self.stick_buffer = []
        self.stick_buffer_size = 5
        self.min_keypoint_confidence = 0.4
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        #mediapipe configuration optimized for cpu-only systems
        #model_complexity=1 balances accuracy and speed on cpu (0 is too inaccurate, 2 is too slow)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  #video mode for continuous tracking
            model_complexity=1,        #general model - better accuracy than lite, still fast on cpu
            min_detection_confidence=0.5,  #higher threshold reduces false detections and jitter
            min_tracking_confidence=0.5,   #balanced with detection for consistent tracking
            smooth_landmarks=True      #temporal smoothing for stable landmarks
        )

        try:
            # Try to load GCN models first
            print("[info] attempting to load GCN specialist models...")
            self.gcn_engine = get_gcn_engine(device=self.device_info.get('torch_device', 'cpu'))
            self.pose_classifier_model = self.gcn_engine
            self.label_encoder = None  # GCN uses internal CLASS_NAMES
            self.scaler = None
            self.is_gcn = True
            self.is_ensemble = False
            print("[info] GCN specialist models loaded successfully")
        except Exception as e:
            print(f"[warning] Could not load GCN models, falling back to legacy: {e}")
            self.is_gcn = False
            self.gcn_engine = None
            
            # Legacy model loading (TensorFlow/sklearn)
            try:
                #try app/models first, then fall back to ml/models
                models_dir = get_resource_path('app/models')
                if not os.path.exists(models_dir):
                    models_dir = get_resource_path('ml/models')
                active_model_file = os.path.join(models_dir, 'active_model.json')
                
                print(f"[debug] checking active_model.json at: {active_model_file}")
                print(f"[debug] file exists: {os.path.exists(active_model_file)}")
                
                #try to load from active_model.json (new versioned system)
                if os.path.exists(active_model_file):
                    import json
                    with open(active_model_file, 'r') as f:
                        active_config = json.load(f)
                    
                    #support both relative and absolute paths (for backwards compatibility)
                    #if path is absolute and exists, use it; otherwise treat as relative
                    model_path = active_config['model_path']
                    encoder_path = active_config['encoder_path']
                    scaler_path = active_config.get('scaler_path')
                    version_name = active_config['version']
                    
                    #convert to resource paths if not absolute or doesn't exist
                    #use models_dir to support both app/models and ml/models locations
                    if not os.path.isabs(model_path) or not os.path.exists(model_path):
                        model_path = os.path.join(models_dir, version_name, os.path.basename(model_path))
                    if not os.path.isabs(encoder_path) or not os.path.exists(encoder_path):
                        encoder_path = os.path.join(models_dir, version_name, os.path.basename(encoder_path))
                    if scaler_path and (not os.path.isabs(scaler_path) or not os.path.exists(scaler_path)):
                        scaler_path = os.path.join(models_dir, version_name, os.path.basename(scaler_path))
                    
                    print(f"[info] using model version: {version_name}")
                    
                    #check if this is an ensemble model first (before checking model.keras)
                    version_path = os.path.join(models_dir, version_name)
                    metadata_path = os.path.join(version_path, 'metadata.json')
                    ensemble_config_path = os.path.join(version_path, 'ensemble_config.json')
                    
                    print(f"[debug] version_path: {version_path}")
                    print(f"[debug] metadata exists: {os.path.exists(metadata_path)}")
                    print(f"[debug] ensemble_config exists: {os.path.exists(ensemble_config_path)}")
                    
                    is_ensemble = False
                    model_type = 'dnn'  #default
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        model_type = metadata.get('model_type', 'dnn')
                        is_ensemble = (model_type == 'ensemble')
                        print(f"[debug] model_type from metadata: {model_type}")
                        print(f"[debug] is_ensemble: {is_ensemble}")
                    
                    if is_ensemble:
                        #load ensemble configuration
                        if os.path.exists(ensemble_config_path):
                            with open(ensemble_config_path, 'r') as f:
                                ensemble_config = json.load(f)
                            
                            print(f"[info] loading ensemble with models: {ensemble_config['model_versions']}")
                            
                            #import ensemble classifier from app utils
                            try:
                                from app.utils.ensemble_model import EnsembleClassifier
                            except ImportError as e:
                                print(f"[error] could not import EnsembleClassifier: {e}")
                                raise
                            
                            #load ensemble with models_dir pointing to app/models
                            self.pose_classifier_model = EnsembleClassifier(
                                model_versions=ensemble_config['model_versions'],
                                voting=ensemble_config['voting'],
                                weights=ensemble_config['weights'],
                                models_dir=models_dir,
                                verbose=False
                            )
                            self.label_encoder = joblib.load(encoder_path)
                            
                            if scaler_path and os.path.exists(scaler_path):
                                self.scaler = joblib.load(scaler_path)
                            else:
                                self.scaler = None
                            
                            self.is_ensemble = True
                            print(f"[info] ensemble model loaded: {', '.join([m.split('_')[0] for m in ensemble_config['model_versions']])}")
                        else:
                            print(f"[error] ensemble_config.json not found at: {ensemble_config_path}")
                            raise FileNotFoundError("ensemble_config.json not found")
                    else:
                        #load regular model (dnn, rf, or xgboost)
                        self.is_ensemble = False
                        
                        if model_type == 'dnn':
                            # load keras model
                            import tensorflow as tf
                            if not os.path.exists(model_path):
                                raise FileNotFoundError("model file not found")
                            self.pose_classifier_model = tf.keras.models.load_model(model_path)
                            if not self.pose_classifier_model.optimizer:
                                self.pose_classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        else:
                            #load rf or xgboost
                            model_joblib = model_path.replace('.keras', '.joblib')
                            if os.path.exists(model_joblib):
                                self.pose_classifier_model = joblib.load(model_joblib)
                            elif os.path.exists(model_path):
                                #try legacy path
                                self.pose_classifier_model = joblib.load(model_path)
                            else:
                                raise FileNotFoundError("model file not found")
                        
                        self.label_encoder = joblib.load(encoder_path)
                        
                        if scaler_path and os.path.exists(scaler_path):
                            self.scaler = joblib.load(scaler_path)
                        else:
                            self.scaler = None
                        
                        print(f"[info] {model_type.upper()} pose classifier loaded")
                else:
                    #fallback to legacy paths
                    model_path = get_resource_path('models/arnis_coordinates_classifier.keras')
                    encoder_path = get_resource_path('models/label_encoder.joblib')
                    scaler_path = get_resource_path('models/scaler.joblib')
                    
                    #if models/ doesn't exist, try ml/models/
                    if not os.path.exists(model_path):
                        print(f"[debug] models/ not found at: {model_path}")
                        model_path = get_resource_path('ml/models/arnis_coordinates_classifier.keras')
                        encoder_path = get_resource_path('ml/models/label_encoder.joblib')
                        scaler_path = get_resource_path('ml/models/scaler.joblib')
                        print(f"[debug] trying ml/models/ at: {model_path}")
                        print(f"[debug] encoder path: {encoder_path}")
                        print(f"[debug] model exists: {os.path.exists(model_path)}")
                        print(f"[debug] encoder exists: {os.path.exists(encoder_path)}")
                        print("[info] using ml/models/ directory")
                    else:
                        print("[info] using legacy model paths")
                    
                    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
                        print(f"[error] model_path exists: {os.path.exists(model_path)}")
                        print(f"[error] encoder_path exists: {os.path.exists(encoder_path)}")
                        print(f"[error] model_path: {model_path}")
                        print(f"[error] encoder_path: {encoder_path}")
                        raise FileNotFoundError("model or encoder not found")

                    #load model with compile=false to handle keras 2.x/3.x compatibility
                    print("[info] loading model (Keras 2.x/3.x compatibility mode)...")
                    import keras
                    self.pose_classifier_model = keras.saving.load_model(model_path, compile=False)
                    
                    #manually compile the model
                    self.pose_classifier_model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    self.label_encoder = joblib.load(encoder_path)
                    self.is_ensemble = False
                    
                    #load scaler if available
                    if scaler_path and os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)
                        print("[info] feature scaler loaded")
                    else:
                        self.scaler = None
                        print("[warning] no scaler found")
                    
                    if not self.pose_classifier_model.optimizer:
                        self.pose_classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    
                    print("[info] keras pose classifier loaded")
            except Exception as legacy_e:
                print(f"[critical] could not load legacy model either: {legacy_e}")
                import traceback
                traceback.print_exc()
                self.pose_classifier_model = None
                self.label_encoder = None
                self.scaler = None
                self.is_ensemble = False
        
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
            
            #run stick detection
            # Lowered confidence threshold to catch fast moving sticks
            results = self.stick_detector(frame, verbose=False, conf=0.25)
            
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
            
            # Find best stick (closest to person if person_bbox provided, otherwise highest conf)
            best_stick_idx = 0
            if person_bbox:
                px1, py1, px2, py2 = person_bbox
                p_area = (px2 - px1) * (py2 - py1)
                best_iou = -1.0
                
                # Check top 3 detections if available
                count = min(3, len(result.boxes))
                for i in range(count):
                    box = result.boxes[i]
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    
                    # Calculate intersection
                    ix1 = max(px1, bx1); iy1 = max(py1, by1)
                    ix2 = min(px2, bx2); iy2 = min(py2, by2)
                    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    
                    # We utilize a modified IoU where we care about intersection with person
                    # Sticks are often held 'out', so simple overlap is sufficient
                    if inter_area > 0:
                        # Prioritize sticks that actually touch the person
                        if i == 0: best_iou = 0.1 # Baseline preference for highest conf
                        if inter_area > best_iou:
                            best_iou = inter_area
                            best_stick_idx = i
            
            #get stick bounding box
            stick_box = result.boxes[best_stick_idx]
            stick_bbox = tuple(map(int, stick_box.xyxy[0].tolist()))
            confidence = stick_box.conf.item()
            
            if debug:
                print(f"[DEBUG-STICK] ✓ Stick detected - Confidence: {confidence:.3f}")
                print(f"[DEBUG-STICK] Stick bbox: {stick_bbox}")
                print(f"[DEBUG-STICK] Stick box class: {stick_box.cls.item() if stick_box.cls is not None else 'None'}")
            
            #get stick keypoints (grip and tip)
            if result.keypoints is not None and len(result.keypoints) > 0:
                if debug:
                    print(f"[DEBUG-STICK] Keypoints object exists, length: {len(result.keypoints)}")
                    print(f"[DEBUG-STICK] Keypoints type: {type(result.keypoints)}")
                
                kpts = result.keypoints[best_stick_idx].data[0] # Use best index
                
                if debug:
                    print(f"[DEBUG-STICK] Keypoints data shape: {kpts.shape if hasattr(kpts, 'shape') else 'N/A'}")
                    print(f"[DEBUG-STICK] Keypoints data: {kpts}")
                
                grip_point = (int(kpts[0][0]), int(kpts[0][1]))
                tip_point = (int(kpts[1][0]), int(kpts[1][1]))
                grip_conf = kpts[0][2].item()
                tip_conf = kpts[1][2].item()
                
                #confidence filtering - lowered to 0.3
                min_kpt_conf = 0.3
                if grip_conf < min_kpt_conf or tip_conf < min_kpt_conf:
                    if debug:
                        print(f"[DEBUG-STICK] low confidence - grip: {grip_conf:.2f}, tip: {tip_conf:.2f}")
                    return None, None
                
                #apply smoothing
                smoothed = self._smooth_stick_keypoints(grip_point, tip_point)
                if smoothed:
                    grip_point, tip_point = smoothed
                
                if debug:
                    print(f"[DEBUG-STICK] grip: {grip_point} (conf: {grip_conf:.2f})")
                    print(f"[DEBUG-STICK] tip: {tip_point} (conf: {tip_conf:.2f})")
                
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

    def process_frame(self, frame, skip_ml_inference=False, skip_stick_detection=False):
        h, w, _ = frame.shape

        #use yolo's built-in bytetrack tracking
        results_yolo = self.yolo_model.track(
            frame, 
            persist=True,  #persist tracks between frames
            tracker="bytetrack.yaml",  #use bytetrack algorithm
            verbose=False, 
            classes=[0],  #person class only
            conf=0.4,      #higher conf for more stable detections (was 0.3)
            imgsz=480      #larger size for better accuracy (was 256)
        )
        
        tracked_persons = []
        for r in results_yolo:
            if r.boxes is not None and r.boxes.id is not None:
                for box, track_id in zip(r.boxes, r.boxes.id):
                    if box.conf[0] >= 0.3:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        tracker_id = int(track_id)
                        
                        #map to stable ids
                        if tracker_id not in self.id_mapping:
                            self.id_mapping[tracker_id] = self.next_stable_id
                            self.next_stable_id += 1
                        stable_id = self.id_mapping[tracker_id]
                        
                        tracked_persons.append([x1, y1, x2, y2, stable_id])
        
        tracked_persons = np.array(tracked_persons) if tracked_persons else np.empty((0, 5))
        
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
                
                #calculate absolute landmarks in frame coordinates
                #landmarks are relative to the crop, so we add the offset
                abs_landmarks = []
                for lm in landmarks_2d:
                    #convert normalized coordinates to crop space, then to frame space
                    abs_x = int(lm.x * crop_w) + offset_x
                    abs_y = int(lm.y * crop_h) + offset_y
                    #clamp to ensure they stay within reasonable bounds
                    abs_x = max(0, min(abs_x, frame.shape[1] - 1))
                    abs_y = max(0, min(abs_y, frame.shape[0] - 1))
                    abs_landmarks.append((abs_x, abs_y, lm.z))
                
                live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks)
                
                # Fallback to 2D angles if 3D failed (ensures we always have feedback data)
                if not live_angles and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                    live_angles = self._calculate_all_angles_2d_dict(pose_results.pose_landmarks.landmark)

                predicted_class, confidence = "N/A", 0.0
                
                #optimization: skip ml inference if requested (use cached from last frame)
                if not skip_ml_inference:
                    if getattr(self, 'is_gcn', False) and self.gcn_engine:
                        try:
                            # 1. Prepare keypoints for GCN (normalized coordinates)
                            landmarks_2d = pose_results.pose_landmarks.landmark
                            pose_kpts_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks_2d])
                            
                            # 2. Get stick keypoints if detected
                            # We'll run stick detection first if it hasn't been run yet for this frame
                            if not skip_stick_detection:
                                stick_res, _ = self._detect_stick_with_yolo(frame, (x1, y1, x2, y2))
                                self._cached_stick_results[person_id] = (stick_res, _)
                            else:
                                stick_res, _ = self._cached_stick_results.get(person_id, (None, None))
                                
                            if stick_res:
                                grip_pt, tip_pt = stick_res
                                h_frame, w_frame = frame.shape[:2]
                                stick_kpts_array = np.array([
                                    [grip_pt[0]/w_frame, grip_pt[1]/h_frame, 0.0, 1.0],
                                    [tip_pt[0]/w_frame, tip_pt[1]/h_frame, 0.0, 1.0]
                                ]).astype(np.float32)
                            else:
                                # Default stick positions [2, 4]
                                stick_kpts_array = np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]).astype(np.float32)
                            
                            # 3. Use modular helper for global features (joint angles, heights, etc.)
                            # Using pose_kpts_array (normalized MediaPipe coordinates)
                            g_feat = compute_global_features_from_kpts(pose_kpts_array, stick_kpts_array)
                            
                            # 4. Run GCN Prediction
                            predicted_class, confidence, _ = self.gcn_engine.predict(
                                pose_kpts_array, stick_kpts_array, g_feat
                            )
                            self._cached_prediction = (predicted_class, confidence)
                        except Exception as e:
                            print(f"[error] GCN inference failed: {e}")
                            pass
                    elif self.pose_classifier_model and self.label_encoder and live_angles:
                        try:
                            world_landmarks = pose_results.pose_world_landmarks.landmark
                            landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
                            hip_center = (landmarks_np[23] + landmarks_np[24]) / 2.0
                            coords = (landmarks_np - hip_center).flatten()
                            
                            #apply scaler if available
                            if self.scaler is not None:
                                coords = self.scaler.transform(coords.reshape(1, -1))[0]
                            
                            #check if ensemble or regular model
                            if getattr(self, 'is_ensemble', False):
                                #ensemble model
                                prediction = self.pose_classifier_model.predict(coords.reshape(1, -1))[0]
                                predicted_class = prediction
                                confidence = 0.85  #placeholder
                            else:
                                #regular model (dnn, rf, xgboost)
                                coords_input = np.expand_dims(coords, axis=0)
                                if hasattr(self.pose_classifier_model, 'predict_proba'):
                                    pred_proba = self.pose_classifier_model.predict_proba(coords_input)[0]
                                    pred_index = np.argmax(pred_proba)
                                    confidence = pred_proba[pred_index]
                                    predicted_class = self.label_encoder.inverse_transform([pred_index])[0]
                                else:
                                    pred_proba = self.pose_classifier_model.predict(coords_input, verbose=0)[0]
                                    pred_index = np.argmax(pred_proba)
                                    confidence = pred_proba[pred_index]
                                    predicted_class = self.label_encoder.inverse_transform([pred_index])[0]
                            
                            self._cached_prediction = (predicted_class, confidence)
                        except Exception:
                            pass
                else:
                    #use cached prediction from previous frame
                    if hasattr(self, '_cached_prediction'):
                        predicted_class, confidence = self._cached_prediction
                    else:
                        #first frame, no cache yet - run inference anyway
                        if self.pose_classifier_model and self.label_encoder and live_angles:
                            try:
                                world_landmarks = pose_results.pose_world_landmarks.landmark
                                landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
                                hip_center = (landmarks_np[23] + landmarks_np[24]) / 2.0
                                coords = (landmarks_np - hip_center).flatten()
                                
                                if self.scaler is not None:
                                    coords = self.scaler.transform(coords.reshape(1, -1))[0]
                                
                                pred_proba = self.pose_classifier_model.predict(np.expand_dims(coords, axis=0), verbose=0)[0]
                                pred_index = np.argmax(pred_proba)
                                confidence = pred_proba[pred_index]
                                predicted_class = self.label_encoder.inverse_transform([pred_index])[0]
                                self._cached_prediction = (predicted_class, confidence)
                            except Exception:
                                pass
                
                #optimization: skip stick detection if requested (use cached from last frame)
                if not skip_stick_detection:
                    if self.debug_stick:
                        print(f"[DEBUG-PROCESS] Calling stick detection for person {person_id}")
                        print(f"[DEBUG-PROCESS] Person bbox: {(x1, y1, x2, y2)}")
                    
                    stick_endpoints, stick_bbox = self._detect_stick_with_yolo(frame, (x1, y1, x2, y2), debug=self.debug_stick)
                    
                    #cache stick detection results
                    self._cached_stick_results[person_id] = (stick_endpoints, stick_bbox)
                    
                    if self.debug_stick:
                        print(f"[DEBUG-PROCESS] Stick detection returned:")
                        print(f"[DEBUG-PROCESS]   stick_endpoints: {stick_endpoints}")
                        print(f"[DEBUG-PROCESS]   stick_bbox: {stick_bbox}")
                else:
                    #use cached stick detection from previous frame
                    if person_id in self._cached_stick_results:
                        stick_endpoints, stick_bbox = self._cached_stick_results[person_id]
                        if self.debug_stick:
                            print(f"[DEBUG-PROCESS] Using cached stick detection for person {person_id}")
                    else:
                        #first frame, no cache yet - run detection anyway
                        stick_endpoints, stick_bbox = self._detect_stick_with_yolo(frame, (x1, y1, x2, y2), debug=self.debug_stick)
                        self._cached_stick_results[person_id] = (stick_endpoints, stick_bbox)
                
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

    def _calculate_all_angles_2d_dict(self, landmarks):
        if not landmarks: return None
        
        # Helper to get coords from list
        def get_coords(idx):
            if idx < len(landmarks):
                return (landmarks[idx].x, landmarks[idx].y) # Ignore Z for 2D calc
            return None

        # Similar set of angles as 3D
        try:
            # MediaPipe indices: 11=L_SH, 12=R_SH, 13=L_ELB, 14=R_ELB, 15=L_WR, 16=R_WR
            # 23=L_HIP, 24=R_HIP, 25=L_KNEE, 26=R_KNEE, 27=L_ANK, 28=R_ANK
            return {
                'left_elbow': self._calculate_angle_2d(get_coords(11), get_coords(13), get_coords(15)),
                'right_elbow': self._calculate_angle_2d(get_coords(12), get_coords(14), get_coords(16)),
                'left_shoulder': self._calculate_angle_2d(get_coords(23), get_coords(11), get_coords(13)),
                'right_shoulder': self._calculate_angle_2d(get_coords(24), get_coords(12), get_coords(14)),
                'left_knee': self._calculate_angle_2d(get_coords(23), get_coords(25), get_coords(27)),
                'right_knee': self._calculate_angle_2d(get_coords(24), get_coords(26), get_coords(28)),
            }
        except Exception: 
            return None

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

    def _smooth_stick_keypoints(self, grip_point, tip_point):
        #add to buffer
        self.stick_buffer.append((grip_point, tip_point))
        
        #keep buffer at max size
        if len(self.stick_buffer) > self.stick_buffer_size:
            self.stick_buffer.pop(0)
        
        #need at least 2 points to smooth
        if len(self.stick_buffer) < 2:
            return grip_point, tip_point
        
        #average all points in buffer
        avg_grip_x = int(np.mean([p[0][0] for p in self.stick_buffer]))
        avg_grip_y = int(np.mean([p[0][1] for p in self.stick_buffer]))
        avg_tip_x = int(np.mean([p[1][0] for p in self.stick_buffer]))
        avg_tip_y = int(np.mean([p[1][1] for p in self.stick_buffer]))
        
        return (avg_grip_x, avg_grip_y), (avg_tip_x, avg_tip_y)
    
    def draw_stick_debug(self, frame, stick_endpoints, stick_bbox=None):
        #draw debug overlay for stick detection
        if stick_endpoints:
            grip, tip = stick_endpoints
            #draw keypoints
            cv2.circle(frame, grip, 8, (0, 255, 0), -1)  #green = grip
            cv2.circle(frame, tip, 8, (0, 0, 255), -1)   #red = tip
            #draw line
            cv2.line(frame, grip, tip, (255, 255, 0), 3)
            #labels
            cv2.putText(frame, "GRIP", (grip[0]-20, grip[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "TIP", (tip[0]-15, tip[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if stick_bbox:
            x1, y1, x2, y2 = stick_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        return frame

    def clear_stick_buffer(self):
        self.stick_buffer = []

    def reset_tracker(self):
        #reset tracking by reinitializing the model (clears bytetrack state)
        self.yolo_model.predictor = None  #clear predictor to reset tracking
        self.track_history = {}
        self.id_mapping = {}
        self.next_stable_id = 1
        print("[info] ByteTrack reset - IDs reinitialized")

    def set_viewpoint(self, viewpoint):
        """Update the active viewpoint for GCN models"""
        if getattr(self, 'is_gcn', False) and self.gcn_engine:
            self.gcn_engine.set_viewpoint(viewpoint)
            print(f"[info] PoseAnalyzer viewpoint updated to: {viewpoint}")

    def close(self):
        self.pose.close()
        print("[info] pose analyzer closed.")