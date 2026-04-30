import os
import sys
import math
import cv2
import numpy as np

import mediapipe as mp
# TensorFlow and Keras will be lazy-loaded in legacy fallback

from ultralytics import YOLO

#import resource path helper for pyinstaller compatibility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.resource_path import get_resource_path
from app.utils.device_manager import configure_device, get_yolo_device

# GCN integration imports
from app.computer_vision.gcn_inference import get_gcn_engine
from app.models.gcn.feature_extraction import compute_global_features_from_kpts
from app.models.gcn.model_architecture import CLASS_NAMES

class PoseAnalyzer:
    def __init__(self, detection_interval=3, stick_model_path=None, debug_stick=False, disable_stick_correction=False):
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
        self.disable_stick_correction = disable_stick_correction
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
        # FIX #6: Separate static-mode Pose for snapshot classification.
        # FIX #6: Static pose detector for lessons/single images (prevents temporal bleed)
        # static_image_mode=True disables temporal smoothing - critical for lesson accuracy
        # model_complexity=2 required for V5 front model (trained on complexity=2 data);
        # using complexity=1 causes feature distribution mismatch and drops accuracy.
        # Video-mode (self.pose) keeps complexity=1 for live camera performance.
        self.pose_static = self.mp_pose.Pose(
            static_image_mode=True,    #each frame treated independently
            model_complexity=2,        #match V5 training data (critical for accuracy)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=False
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
            print(f"[ERROR] Could not load GCN models: {e}")
            self.is_gcn = False
            self.gcn_engine = None
            # No fallback to legacy models
            print("[CRITICAL] GCN models failed to load. Pose classification will be unavailable.")

        # Load YOLO-Pose for fast countdown visualization
        try:
            print("[info] loading YOLOv8n-Pose for countdown visualization...")
            self.yolo_pose = YOLO('yolov8n-pose.pt')
            print("[info] YOLOv8n-Pose loaded successfully")
        except Exception as e:
            print(f"[warning] Could not load YOLO-Pose: {e}")
            self.yolo_pose = None

        self.detection_interval = detection_interval
        self.frame_count = 0
        self.last_detections = []
        print("[info] computer vision components ready.")

    def clear_session_cache(self):
        """Clear all per-session inference caches.
        
        Call this at the start of every new repetition / session so that
        stale predictions, stick results and global features from the
        previous rep cannot bleed into the next one.
        """
        # GCN prediction cache
        if hasattr(self, '_cached_prediction'):
            del self._cached_prediction
        if hasattr(self, '_cached_g_feat'):
            del self._cached_g_feat
        # Stick detection result cache (keyed by person_id)
        self._cached_stick_results.clear()
        # Stick keypoint smoothing buffer
        self.stick_buffer.clear()
        print("[PoseAnalyzer] Session cache cleared.")

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _detect_stick_with_yolo(self, frame, person_bbox=None, debug=False, skip_smoothing=False):
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
            # Lowered confidence threshold to catch partially occluded sticks
            results = self.stick_detector(frame, verbose=False, conf=0.15)
            
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
                
                # NOTE: confidence filtering relaxed to match deployment_package
                # behavior. The v5 model was trained/validated with raw YOLO
                # outputs (no confidence thresholding). Filtering at inference
                # causes distribution mismatch on partially-occluded sticks.
                min_grip_conf = 0.0
                min_tip_conf = 0.0
                if grip_conf < min_grip_conf or tip_conf < min_tip_conf:
                    if debug:
                        print(f"[DEBUG-STICK] low confidence - grip: {grip_conf:.2f} (min: {min_grip_conf}), tip: {tip_conf:.2f} (min: {min_tip_conf})")
                    return None, None
                
                #apply smoothing (only for video mode, not snapshot)
                if not skip_smoothing:
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

    def process_frame(self, frame, skip_ml_inference=False, skip_stick_detection=False, mode='snapshot', target_pose=None, stick_hand_config=None, skip_threshold=False, use_crop=False):
        """
        Process frame with mode-based detection strategy:
        
        MODE parameter controls MediaPipe temporal smoothing behavior:
        - 'snapshot' uses pose_static (static_image_mode=True, no temporal smoothing)
        - other values use pose (video mode with temporal smoothing)
        
        Use 'snapshot' for lessons, classification, and single images.
        Use 'video' or other values for live camera streams.
        
        Valid modes: 'snapshot', 'video', 'countdown'
        
        SNAPSHOT MODE (GCN Classification):
        - Uses YOLO regular detection (.predict()) - no tracking needed
        - Gets person bounding boxes → MediaPipe 33 keypoints → GCN classification
        - Used for: Test images, kiosk snapshot capture
        
        COUNTDOWN MODE (Live Visualization):
        - Uses YOLO-Pose with ByteTrack (.track()) - maintains stable IDs across frames
        - Gets 17 COCO keypoints directly from YOLO-Pose
        - Used for: Real-time countdown visualization, no classification needed
        
        Args:
            frame: Input image frame
            skip_ml_inference: If True, skip GCN classification (just get landmarks)
            skip_stick_detection: If True, skip stick detection
            mode: 'snapshot' (static frame, GCN) or 'countdown' (video, tracking)
            target_pose: Optional target pose name (for batch testing with hand override)
            stick_hand_config: Optional dict mapping pose names to hand preferences {'pose_name': {'hand': 'left'/'right', ...}}
        """
        # Validate mode parameter (D5)
        if mode not in ('snapshot', 'video', 'countdown'):
            raise ValueError(f"Invalid mode: {mode}. Must be 'snapshot', 'video', or 'countdown'")
        
        h, w, _ = frame.shape
        
        print(f"[DEBUG-YOLO] Processing frame: {w}x{h}, mode={mode}")

        # DETECTION STRATEGY BASED ON MODE
        if mode == 'snapshot':
            # Regular detection for static images (GCN classification)
            print(f"[DEBUG-YOLO] Using regular detection for snapshot mode (GCN)")
            results_yolo = self.yolo_model.predict(
                frame,
                verbose=False,
                classes=[0],  # person class only
                conf=0.3,     # detection confidence
                imgsz=480
            )
        else:
            # ByteTrack tracking for live video (countdown/visualization)
            print(f"[DEBUG-YOLO] Using ByteTrack tracking for {mode} mode (visualization)")
            results_yolo = self.yolo_model.track(
                frame, 
                persist=False,  # DISABLED: persist=True causes ghost tracks when processing sequential zones
                tracker="bytetrack.yaml",  # use bytetrack algorithm
                verbose=False, 
                classes=[0],  # person class only
                conf=0.4,     # higher conf for more stable detections (was 0.3)
                imgsz=480     # larger size for better accuracy (was 256)
            )
        
        print(f"[DEBUG-YOLO] YOLO results count: {len(results_yolo)}")
        
        tracked_persons = []
        for r in results_yolo:
            if r.boxes is not None:
                print(f"[DEBUG-YOLO] Boxes found: {len(r.boxes)}")
                
                # For snapshot mode, use simple indexing (no tracking IDs)
                if mode == 'snapshot':
                    for idx, box in enumerate(r.boxes):
                        if box.conf[0] >= 0.3:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            print(f"[DEBUG-YOLO] Person {idx} detected: bbox=({x1},{y1},{x2},{y2}), conf={box.conf[0]:.2f}")
                            # Use simple index as ID for snapshot mode
                            tracked_persons.append([x1, y1, x2, y2, idx])
                else:
                    # For tracking modes, use track IDs
                    if r.boxes.id is not None:
                        print(f"[DEBUG-YOLO] IDs found: {len(r.boxes.id)}")
                        for box, track_id in zip(r.boxes, r.boxes.id):
                            if box.conf[0] >= 0.3:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                tracker_id = int(track_id)
                                print(f"[DEBUG-YOLO] Person detected: bbox=({x1},{y1},{x2},{y2}), conf={box.conf[0]:.2f}, track_id={tracker_id}")
                                
                                # Map to stable ids
                                if tracker_id not in self.id_mapping:
                                    self.id_mapping[tracker_id] = self.next_stable_id
                                    self.next_stable_id += 1
                                stable_id = self.id_mapping[tracker_id]
                                
                                tracked_persons.append([x1, y1, x2, y2, stable_id])
                    else:
                        print(f"[DEBUG-YOLO] No IDs in this result (tracking not initialized)")
            else:
                print(f"[DEBUG-YOLO] No boxes in this result")
        
        print(f"[DEBUG-YOLO] Total tracked persons: {len(tracked_persons)}")
        
        tracked_persons = np.array(tracked_persons) if tracked_persons else np.empty((0, 5))
        
        analysis_results = { 
            int(p[4]): {
                'id': int(p[4]), 'bbox': tuple(map(int, p[:4])), 
                'predicted_class': "N/A", 'confidence': 0.0, 
                'live_angles': None, 'landmarks': None, 
                'stick_endpoints': None, 'stick_keypoints': None, 'grip_angle': None,
                'stick_foreshortened': False
            } for p in tracked_persons 
        }

        # --- MEDIAPIPE SETUP ---
        pose_instance = self.pose_static if mode == 'snapshot' else self.pose
        if mode == 'snapshot':
            assert pose_instance is self.pose_static, "Must use pose_static for snapshot mode"
        
        # Full-frame MediaPipe (run once, shared across all persons when not cropping)
        h_frame, w_frame = frame.shape[:2]
        full_frame_pose_results = None
        has_full_frame_pose = False
        if not use_crop:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_frame_pose_results = pose_instance.process(frame_rgb)
            has_full_frame_pose = full_frame_pose_results and full_frame_pose_results.pose_landmarks
            print(f"[DEBUG-MediaPipe] Full-frame pose: detected={has_full_frame_pose}")
        
        for person in tracked_persons:
            person_id = int(person[4])
            x1, y1, x2, y2 = map(int, person[:4])
            
            # MODE-BASED POSE PROCESSING
            # Countdown: Use YOLO-Pose for fast visualization (17 keypoints)
            # Snapshot: Use MediaPipe for accurate classification (33 keypoints)
            
            if mode == 'countdown' and self.yolo_pose is not None:
                try:
                    yolo_results = self.yolo_pose(frame, verbose=False)
                    if len(yolo_results) > 0 and yolo_results[0].keypoints is not None:
                        keypoints = yolo_results[0].keypoints.data
                        if len(keypoints) > 0:
                            kpts = keypoints[0].cpu().numpy()
                            abs_landmarks = []
                            for kpt in kpts:
                                x, y, conf = kpt
                                if conf > 0.5:
                                    abs_x = int(x)
                                    abs_y = int(y)
                                    abs_x = max(0, min(abs_x, w_frame - 1))
                                    abs_y = max(0, min(abs_y, h_frame - 1))
                                    abs_landmarks.append((abs_x, abs_y, 0.0))
                                else:
                                    abs_landmarks.append((-1, -1, 0.0))
                            
                            analysis_results[person_id]['landmarks_absolute'] = abs_landmarks
                            analysis_results[person_id]['landmarks_2d'] = None
                            analysis_results[person_id]['live_angles'] = None
                            
                            if not skip_stick_detection:
                                stick_endpoints, stick_bbox = self._detect_stick_with_yolo(frame, (x1, y1, x2, y2), debug=self.debug_stick)
                                if stick_endpoints:
                                    analysis_results[person_id]['stick_endpoints'] = stick_endpoints
                                    grip_pt, tip_pt = stick_endpoints
                                    analysis_results[person_id]['stick_keypoints'] = {'grip': grip_pt, 'tip': tip_pt}
                            
                            continue
                except Exception as e:
                    print(f"[warning] YOLO-Pose failed, falling back to MediaPipe: {e}")
                    # Fall through to MediaPipe
            
            # --- MediaPipe pose detection: full-frame vs per-person crop ---
            pose_results = None
            landmarks_2d = None
            abs_landmarks = []
            
            if use_crop:
                # PER-PERSON CROP PATH (restored for A/B testing)
                x1_pad = max(0, x1 - 20)
                y1_pad = max(0, y1 - 20)
                x2_pad = min(w_frame, x2 + 20)
                y2_pad = min(h_frame, y2 + 20)
                person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if person_crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_results = pose_instance.process(crop_rgb)
                if pose_results and pose_results.pose_landmarks:
                    landmarks_2d = pose_results.pose_landmarks.landmark
                    crop_h, crop_w = person_crop.shape[:2]
                    for lm in landmarks_2d:
                        if hasattr(lm, 'visibility') and lm.visibility < 0.3:
                            abs_landmarks.append((-1, -1, 0.0))
                            continue
                        abs_x = int(lm.x * crop_w) + x1_pad
                        abs_y = int(lm.y * crop_h) + y1_pad
                        abs_x = max(0, min(abs_x, w_frame - 1))
                        abs_y = max(0, min(abs_y, h_frame - 1))
                        abs_landmarks.append((abs_x, abs_y, lm.z))
            else:
                # FULL-FRAME PATH (default, matches v5 training)
                if has_full_frame_pose:
                    pose_results = full_frame_pose_results
                    landmarks_2d = pose_results.pose_landmarks.landmark
                    for lm in landmarks_2d:
                        if hasattr(lm, 'visibility') and lm.visibility < 0.3:
                            abs_landmarks.append((-1, -1, 0.0))
                            continue
                        abs_x = int(lm.x * w_frame)
                        abs_y = int(lm.y * h_frame)
                        abs_x = max(0, min(abs_x, w_frame - 1))
                        abs_y = max(0, min(abs_y, h_frame - 1))
                        abs_landmarks.append((abs_x, abs_y, lm.z))
            
            if not (pose_results and landmarks_2d):
                continue
            
            live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks)
            if not live_angles and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                live_angles = self._calculate_all_angles_2d_dict(pose_results.pose_landmarks.landmark)
            
            # Build pose_kpts_array normalized to full frame
            if use_crop:
                crop_h, crop_w = person_crop.shape[:2]
                pose_kpts_array = np.array([
                    [lm.x * crop_w / w_frame,
                     lm.y * crop_h / h_frame,
                     lm.z,
                     lm.visibility]
                    for lm in landmarks_2d
                ])
            else:
                pose_kpts_array = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in landmarks_2d
                ])
            
            predicted_class, confidence = "N/A", 0.0
            
            if not skip_ml_inference:
                print(f"[DEBUG-GCN] skip_ml_inference=False, checking GCN...")
                print(f"[DEBUG-GCN] is_gcn={getattr(self, 'is_gcn', False)}, gcn_engine exists={self.gcn_engine is not None}")
                if getattr(self, 'is_gcn', False) and self.gcn_engine:
                    path_label = "CROP" if use_crop else "FULL-FRAME"
                    print(f"[DEBUG-GCN] ✓ Entering GCN inference block (V5 {path_label})...")
                    try:
                        # 1. Stick detection
                        if not skip_stick_detection:
                            is_snapshot = (mode == 'snapshot')
                            print(f"[DEBUG-STICK-CALL] Calling _detect_stick_with_yolo for person {person_id}...")
                            stick_res, _ = self._detect_stick_with_yolo(frame, (x1, y1, x2, y2), skip_smoothing=is_snapshot, debug=True)
                            print(f"[DEBUG-STICK-CALL] Result: {stick_res}")
                            
                            # Validate stick result against wrists
                            if stick_res and landmarks_2d:
                                grip_pt, _ = stick_res
                                mp_lm = self.mp_pose.PoseLandmark
                                l_wrist = landmarks_2d[mp_lm.LEFT_WRIST]
                                r_wrist = landmarks_2d[mp_lm.RIGHT_WRIST]
                                if use_crop:
                                    lx = int(l_wrist.x * crop_w) + x1_pad
                                    ly = int(l_wrist.y * crop_h) + y1_pad
                                    rx = int(r_wrist.x * crop_w) + x1_pad
                                    ry = int(r_wrist.y * crop_h) + y1_pad
                                else:
                                    lx = int(l_wrist.x * w_frame)
                                    ly = int(l_wrist.y * h_frame)
                                    rx = int(r_wrist.x * w_frame)
                                    ry = int(r_wrist.y * h_frame)
                                l_dist = np.hypot(grip_pt[0] - lx, grip_pt[1] - ly)
                                r_dist = np.hypot(grip_pt[0] - rx, grip_pt[1] - ry)
                                valid_thresh = w_frame * 0.25
                                if min(l_dist, r_dist) > valid_thresh:
                                    print(f"[DEBUG-STICK] Discarding stick - too far from wrists. Grip: {grip_pt}, Wrists: {(lx, ly)}, {(rx, ry)}")
                                    stick_res = None
                                
                                self._cached_stick_results[person_id] = (stick_res, _)
                        else:
                            stick_res, _ = self._cached_stick_results.get(person_id, (None, None))
                        
                        if stick_res:
                            if self.disable_stick_correction:
                                if self.debug_stick:
                                    print(f"[DEBUG-PROCESS] Using RAW YOLO stick detection (correction disabled)")
                                stick_endpoints = stick_res
                                analysis_results[person_id]['stick_endpoints'] = stick_endpoints
                            else:
                                stick_endpoints = stick_res
                                analysis_results[person_id]['stick_endpoints'] = stick_endpoints
                                grip_pt, tip_pt = stick_endpoints
                                analysis_results[person_id]['stick_keypoints'] = {'grip': grip_pt, 'tip': tip_pt}
                                
                                dx = tip_pt[0] - grip_pt[0]
                                dy = tip_pt[1] - grip_pt[1]
                                angle = math.degrees(math.atan2(-dy, dx))
                                analysis_results[person_id]['grip_angle'] = angle
                                
                                raw_yolo_length = np.linalg.norm(np.array(tip_pt) - np.array(grip_pt))
                                FORESHORTEN_THRESHOLD_PX = 40
                                analysis_results[person_id]['stick_foreshortened'] = raw_yolo_length < FORESHORTEN_THRESHOLD_PX
                        else:
                            stick_endpoints = None
                            analysis_results[person_id]['stick_endpoints'] = None
                            analysis_results[person_id]['stick_keypoints'] = None
                            analysis_results[person_id]['grip_angle'] = None
                            analysis_results[person_id]['stick_foreshortened'] = False
                        
                        # 2. Build stick keypoints array (full-frame normalized)
                        if stick_endpoints:
                            grip_pt, tip_pt = stick_endpoints
                            stick_kpts = np.array([
                                [grip_pt[0] / w_frame, grip_pt[1] / h_frame, 0.0, 1.0],
                                [tip_pt[0] / w_frame, tip_pt[1] / h_frame, 0.0, 1.0]
                            ])
                            has_stick_detected = True
                        else:
                            stick_kpts = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
                            has_stick_detected = False
                        
                        # 3. Compute global features
                        from app.models.gcn.feature_extraction import compute_global_features_from_kpts
                        global_features = compute_global_features_from_kpts(
                            pose_kpts_array, stick_kpts,
                            world_landmarks=pose_results.pose_world_landmarks.landmark if pose_results.pose_world_landmarks else None,
                            has_stick_detected=has_stick_detected,
                            version='v5'
                        )
                        
                        # 4. Run GCN classification
                        predicted_class, confidence, _ = self.gcn_engine.predict(
                            pose_kpts_array, stick_kpts, global_features
                        )
                        print(f"[DEBUG-GCN] V5 result: {predicted_class} @ {confidence:.4f}")
                    except Exception as e:
                        print(f"[ERROR] GCN inference failed (V5): {e}")
                        import traceback
                        traceback.print_exc()
            
            # Store results
            analysis_results[person_id]['landmarks'] = pose_kpts_array
            analysis_results[person_id]['landmarks_absolute'] = abs_landmarks
            analysis_results[person_id]['live_angles'] = live_angles
            analysis_results[person_id]['predicted_class'] = predicted_class
            analysis_results[person_id]['confidence'] = confidence

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

    def draw_visual_cues(self, frame, feedback, landmarks):
        """
        Draw visual cues (arrows) based on feedback corrections.
        
        Args:
            frame: The image frame to draw on.
            feedback: The feedback dictionary from FeedbackAnalyzer.
            landmarks: List of (x, y, z) absolute coordinates.
        """
        if not feedback or 'corrections' not in feedback:
            return

        for correction in feedback['corrections']:
            joint_name = correction['joint']
            action = correction['action']
            
            # Get associated keypoints for the joint
            points = self._get_joint_keypoints(joint_name, landmarks)
            if not points:
                continue
                
            p_start, p_vertex, p_end = points
            
            # Determine arrow parameters
            start_point = (int(p_end[0]), int(p_end[1]))
            
            # Simple heuristic for direction
            # For extension: Arrow points away from the vertex (along the limb vector)
            # For flexion: Arrow points towards the vertex (along the limb vector reversed)
            
            # Vector v = p_end - p_vertex
            vx = p_end[0] - p_vertex[0]
            vy = p_end[1] - p_vertex[1]
            mag = np.hypot(vx, vy)
            if mag < 1e-6: continue
            
            # Normalize
            vx, vy = vx / mag, vy / mag
            
            arrow_len = 40
            color = (0, 255, 0) # Green for extension/correction
            
            if action in ['extend', 'extend_grip']:
                # Point outward
                end_point = (int(start_point[0] + vx * arrow_len), int(start_point[1] + vy * arrow_len))
            elif action in ['flex', 'retract_grip']:
                # Point inward
                end_point = (int(start_point[0] - vx * arrow_len), int(start_point[1] - vy * arrow_len))
                color = (0, 165, 255) # Orange for flexion
            else:
                continue
                
            cv2.arrowedLine(frame, start_point, end_point, color, 3, tipLength=0.3)
            # Draw a small circle at the start to anchor the arrow
            # cv2.circle(frame, start_point, 3, color, -1)

    def _get_joint_keypoints(self, joint_name, landmarks):
        """Returns (start, vertex, end) points for a joint angle."""
        try:
            mp_lm = self.mp_pose.PoseLandmark
            lm_idx = {
                'left_elbow': (mp_lm.LEFT_SHOULDER, mp_lm.LEFT_ELBOW, mp_lm.LEFT_WRIST),
                'right_elbow': (mp_lm.RIGHT_SHOULDER, mp_lm.RIGHT_ELBOW, mp_lm.RIGHT_WRIST),
                'left_shoulder': (mp_lm.LEFT_HIP, mp_lm.LEFT_SHOULDER, mp_lm.LEFT_ELBOW),
                'right_shoulder': (mp_lm.RIGHT_HIP, mp_lm.RIGHT_SHOULDER, mp_lm.RIGHT_ELBOW),
                'left_knee': (mp_lm.LEFT_HIP, mp_lm.LEFT_KNEE, mp_lm.LEFT_ANKLE),
                'right_knee': (mp_lm.RIGHT_HIP, mp_lm.RIGHT_KNEE, mp_lm.RIGHT_ANKLE),
                'wrist': (mp_lm.RIGHT_ELBOW, mp_lm.RIGHT_WRIST, mp_lm.RIGHT_INDEX) # simplified for grip
            }
            
            # For grip/wrist, we might need a workaround if it's not a standard angle
            if joint_name == 'wrist':
               # Just return None for now or map to a generic limb
               # Or use the specific logic if we tracked the stick. 
               # Since 'wrist' was passed for grip corrections, let's use the hand.
               # If specific side isn't known, might be tricky. 
               # FeedbackAnalyzer generalized it to 'wrist'. 
               # Let's check which hand is holding the stick if possible, or defaulting to right for now.
               indices = lm_idx['right_elbow'] # Use right arm for generic 'wrist' corrections if stick is assumed right
               pass
            
            if joint_name not in lm_idx:
                return None
                
            idx_start, idx_vertex, idx_end = lm_idx[joint_name]
            
            p_start = landmarks[idx_start]
            p_vertex = landmarks[idx_vertex]
            p_end = landmarks[idx_end]
            
            # Check visibility or bounds (assuming absolute coords are valid if passed)
            return p_start, p_vertex, p_end
            
        except IndexError:
            return None

    def close(self):
        self.pose.close()
        self.pose_static.close()
        print("[info] pose analyzer closed.")