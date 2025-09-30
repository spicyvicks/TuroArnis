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
    def __init__(self, image_width=1920, image_height=1080):
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt')
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.mp_pose = mp.solutions.pose
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
            enable_segmentation=False
        )
        
        self.image_width = image_width
        self.image_height = image_height
        self.mp_drawing = mp.solutions.drawing_utils

        try:
            self.classifier_model = tf.keras.models.load_model('models_tf/arnis_tf_classifier.h5')
            self.label_encoder = joblib.load('models_tf/label_encoder.joblib')
            print("[info] tensorflow classifier model and label encoder loaded.")
        except Exception as e:
            self.classifier_model = None
            self.label_encoder = None
            print(f"[error] could not load tensorflow model or encoder: {e}")
        
        print("[info] computer vision components ready.")

    def process_frame(self, frame, limb_feedback=None):
        """
        processes a video frame to detect, track, and analyze poses.
        
        args:
            frame: input video frame.
            limb_feedback: a dictionary mapping person_id to their limb feedback.
                           e.g., {1: {'left_elbow': 'green', 'right_shoulder': 'red'}}
        
        returns:
            tuple of (annotated frame, list of analysis results)
        """
        if limb_feedback is None:
            limb_feedback = {}
            
        analysis_results = []
        
        # 1. detect and track persons
        results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0])
        detections = np.empty((0, 5))
        for r in results_yolo:
            for box in r.boxes:
                if box.conf[0] >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, box.conf[0]])))
        
        tracked_persons = self.tracker.update(detections)

        # 2. process each tracked person
        for person in tracked_persons:
            x1, y1, x2, y2, person_id = map(int, person)
            
            person_result = {
                'id': person_id, 'bbox': (x1, y1, x2, y2),
                'predicted_class': None, 'confidence': 0.0, 'live_angles': None
            }
            
            # draw the generic user id and box first
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"User {person_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                analysis_results.append(person_result)
                continue

            image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(image_rgb)
            
            if pose_results.pose_landmarks:
                # get the specific feedback for this person, if any
                feedback_for_person = limb_feedback.get(person_id, {})
                self._draw_landmarks_on_main_frame(frame, pose_results, x1, y1, x2, y2, feedback_for_person)
            
            if self.classifier_model and self.label_encoder and pose_results.pose_world_landmarks:
                try:
                    live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks.landmark)
                    person_result['live_angles'] = live_angles

                    if live_angles:
                        live_features = pd.DataFrame([list(live_angles.values())])
                        prediction_proba = self.classifier_model.predict(live_features)[0]
                        prediction_index = np.argmax(prediction_proba)
                        
                        person_result['confidence'] = prediction_proba[prediction_index]
                        person_result['predicted_class'] = self.label_encoder.inverse_transform([prediction_index])[0]
                
                except Exception as e:
                    print(f"error during tensorflow analysis for user {person_id}: {e}")

            analysis_results.append(person_result)

        return frame, analysis_results

    # --- (helper methods for angle calculation are unchanged) ---
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

    # --- new: enhanced landmark drawing function ---
    def _draw_landmarks_on_main_frame(self, main_frame, pose_results, x1, y1, x2, y2, feedback):
        if not pose_results.pose_landmarks:
            return

        # define colors
        colors = {
            'default_landmark': (245, 117, 66), # blue
            'default_connection': (245, 66, 230), # purple
            'red': (0, 0, 255),
            'green': (0, 255, 0)
        }

        # create a deep copy of the landmarks to modify them
        landmarks_copy = landmark_pb2.NormalizedLandmarkList()
        landmarks_copy.CopyFrom(pose_results.pose_landmarks)

        # translate coordinates
        frame_height, frame_width, _ = main_frame.shape
        crop_width, crop_height = x2 - x1, y2 - y1
        for landmark in landmarks_copy.landmark:
            pixel_x = landmark.x * crop_width + x1
            pixel_y = landmark.y * crop_height + y1
            landmark.x, landmark.y = pixel_x / frame_width, pixel_y / frame_height

        # draw connections first with default color
        self.mp_drawing.draw_landmarks(
            main_frame, landmarks_copy, self.mp_pose.POSE_CONNECTIONS,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=colors['default_connection'], thickness=2)
        )
        
        # now draw landmarks, coloring them based on feedback
        for idx, landmark in enumerate(landmarks_copy.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name.lower()
            
            # determine the color for this specific landmark
            color_name = feedback.get(landmark_name, 'default_landmark')
            landmark_color = colors.get(color_name, colors['default_landmark'])

            # draw the individual landmark
            center_coordinates = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, frame_width, frame_height)
            if center_coordinates:
                cv2.circle(main_frame, center_coordinates, 5, landmark_color, -1)


    def close(self):
        self.pose.close()
        print("[info] pose analyzer closed.")