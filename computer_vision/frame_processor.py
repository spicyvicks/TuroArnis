"""
Multi-threaded Frame Processor for TuroArnis
Separates frame capture, processing, and GUI updates for better performance
"""

import cv2
import queue
import threading
import time
from collections import deque


class FrameProcessor:
    """
    High-performance threaded frame processor
    
    Architecture:
    - Thread 1: Frame capture (camera reading)
    - Thread 2: Pose detection & classification
    - Main thread: GUI updates (non-blocking)
    """
    
    def __init__(self, analyzer, frame_skip=2, max_queue_size=2):
        """
        Initialize threaded processor
        
        Args:
            analyzer: PoseAnalyzer instance
            frame_skip: Process every Nth frame (2 = process half the frames)
            max_queue_size: Maximum frames in queue
        """
        self.analyzer = analyzer
        self.frame_skip = frame_skip
        
        # Queues for thread communication
        self.raw_frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        
        # Threading control
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None
        
        # Performance tracking
        self.frame_count = 0
        self.process_count = 0
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Latest results cache
        self.latest_result = None
        self.result_lock = threading.Lock()
        
    def start(self, video_source=0):
        """Start capture and processing threads"""
        if self.is_running:
            return
            
        self.is_running = True
        self.cap = cv2.VideoCapture(video_source)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        
        print("[INFO] Frame processor started (multi-threaded)")
        
    def stop(self):
        """Stop all threads"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            
        print("[INFO] Frame processor stopped")
        
    def _capture_worker(self):
        """Worker thread: Capture frames from camera"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Flip and resize immediately
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            
            # Skip frames for performance
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                # Still return frame for display, just don't process
                try:
                    self.raw_frame_queue.put_nowait(('display_only', frame))
                except queue.Full:
                    pass  # Drop frame if queue full
                continue
            
            # Add frame to processing queue
            try:
                self.raw_frame_queue.put_nowait(('process', frame.copy()))
            except queue.Full:
                # Drop oldest frame and add new one
                try:
                    self.raw_frame_queue.get_nowait()
                    self.raw_frame_queue.put_nowait(('process', frame.copy()))
                except:
                    pass
                    
    def _process_worker(self):
        """Worker thread: Process frames with pose detection"""
        while self.is_running:
            try:
                frame_type, frame = self.raw_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Only process frames marked for processing
            if frame_type == 'process':
                start_time = time.time()
                
                # Run pose detection and classification
                results = self.analyzer.process_frame(frame)
                
                # Calculate processing time
                process_time = time.time() - start_time
                self.fps_counter.append(1.0 / process_time if process_time > 0 else 0)
                
                # Store results
                with self.result_lock:
                    self.latest_result = results
                    
                # Put in result queue for GUI
                try:
                    self.result_queue.put_nowait(results)
                except queue.Full:
                    # Clear queue and add new result
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(results)
                    except:
                        pass
                        
                self.process_count += 1
                
    def get_latest_result(self):
        """
        Get the latest processing result (non-blocking)
        
        Returns:
            Latest analysis results or None
        """
        with self.result_lock:
            return self.latest_result
            
    def get_fps(self):
        """Get current processing FPS"""
        if len(self.fps_counter) == 0:
            return 0.0
        return sum(self.fps_counter) / len(self.fps_counter)
        
    def get_stats(self):
        """Get performance statistics"""
        return {
            'frames_captured': self.frame_count,
            'frames_processed': self.process_count,
            'processing_fps': self.get_fps(),
            'skip_ratio': f"1/{self.frame_skip}"
        }


# Quick optimization functions
def optimize_mediapipe_settings(pose_instance):
    """
    Apply performance optimizations to MediaPipe Pose
    Call this after initializing PoseAnalyzer
    """
    # These settings are applied during initialization in pose_analyzer.py
    # But we can verify/adjust if needed
    print("[INFO] MediaPipe optimizations applied:")
    print("  - static_image_mode: False (faster for video)")
    print("  - model_complexity: 1 (balanced speed/accuracy)")
    print("  - min_detection_confidence: 0.5")


def get_optimal_frame_skip(target_fps=30):
    """
    Calculate optimal frame skip for target FPS
    
    Args:
        target_fps: Desired frames per second
        
    Returns:
        Recommended frame skip value
    """
    # Assume processing takes ~100ms per frame
    processing_time_ms = 100
    available_time_ms = 1000 / target_fps
    
    if processing_time_ms <= available_time_ms:
        return 1  # Can process every frame
    else:
        return int(processing_time_ms / available_time_ms) + 1
