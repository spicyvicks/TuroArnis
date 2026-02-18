"""
User Tracking Test Script
Tests ByteTrack multi-person tracking with YOLO person detection on MP4 videos

This script helps verify that the person tracking system works correctly:
- Detects multiple people in frame
- Assigns unique IDs to each person
- Maintains ID consistency across frames
- Handles people entering/leaving the frame

Usage:
    python test_user_tracking.py --video path/to/video.mp4
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict, deque
import time

# Import YOLO
from ultralytics import YOLO


class UserTrackingTester:
    """Test multi-person tracking with ByteTrack"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        print(f"[INFO] Initializing User Tracking Tester")
        print(f"[INFO] Confidence Threshold: {conf_threshold}")
        
        # Load YOLO model with tracking
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Tracking statistics
        self.reset_metrics()
        
        # Color palette for different track IDs
        self.colors = self._generate_colors(50)
        
    def _generate_colors(self, n: int) -> list:
        """Generate distinct colors for track IDs"""
        np.random.seed(42)
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def reset_metrics(self):
        """Reset tracking metrics"""
        self.metrics = {
            'total_frames': 0,
            'frames_with_detections': 0,
            'total_detections': 0,
            'unique_track_ids': set(),
            'track_id_history': defaultdict(list),  # track_id -> list of frame numbers
            'people_per_frame': [],
            'id_switches': 0,
            'lost_tracks': 0,
            'new_tracks': 0,
        }
        self.previous_track_ids = set()
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        """Process a single frame with tracking"""
        # Run YOLO tracking (ByteTrack is built into Ultralytics)
        results = self.model.track(
            frame,
            persist=True,  # Enable tracking across frames
            classes=[0],   # Only track persons (class 0)
            conf=self.conf_threshold,
            verbose=False
        )
        
        detections = []
        current_track_ids = set()
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                
                # Get track ID if available
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                    current_track_ids.add(track_id)
                    self.metrics['unique_track_ids'].add(track_id)
                    self.metrics['track_id_history'][track_id].append(frame_idx)
                
                detections.append({
                    'box': box,
                    'confidence': float(conf),
                    'track_id': track_id
                })
        
        # Update metrics
        self.metrics['total_frames'] += 1
        if detections:
            self.metrics['frames_with_detections'] += 1
            self.metrics['total_detections'] += len(detections)
        self.metrics['people_per_frame'].append(len(detections))
        
        # Detect ID switches and track changes
        new_ids = current_track_ids - self.previous_track_ids
        lost_ids = self.previous_track_ids - current_track_ids
        
        if new_ids:
            self.metrics['new_tracks'] += len(new_ids)
        if lost_ids:
            self.metrics['lost_tracks'] += len(lost_ids)
        
        self.previous_track_ids = current_track_ids
        
        return {
            'detections': detections,
            'num_people': len(detections),
            'track_ids': list(current_track_ids)
        }
    
    def draw_tracking(self, frame: np.ndarray, result: dict, frame_idx: int) -> np.ndarray:
        """Draw tracking visualization on frame"""
        display_frame = frame.copy()
        
        # Draw each detection
        for det in result['detections']:
            x1, y1, x2, y2 = map(int, det['box'])
            conf = det['confidence']
            track_id = det['track_id']
            
            # Choose color based on track ID
            if track_id is not None:
                color = self.colors[track_id % len(self.colors)]
                label = f"ID:{track_id} {conf:.2f}"
            else:
                color = (128, 128, 128)
                label = f"? {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame info
        info_text = [
            f"Frame: {frame_idx}",
            f"People: {result['num_people']}",
            f"Active IDs: {sorted(result['track_ids'])}",
            f"Total Unique IDs: {len(self.metrics['unique_track_ids'])}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(display_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        return display_frame
    
    def test_video(self, video_path: str, max_frames: int = None,
                   display: bool = True, save_output: bool = False) -> dict:
        """Test tracking on a video file"""
        print(f"\n[INFO] Testing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video FPS: {fps:.2f}")
        print(f"[INFO] Total frames: {total_frames}")
        print(f"[INFO] Resolution: {width}x{height}")
        
        if max_frames:
            print(f"[INFO] Processing first {max_frames} frames")
        
        # Setup video writer if saving
        writer = None
        if save_output:
            output_path = Path(video_path).stem + '_tracked.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[INFO] Saving output to: {output_path}")
        
        self.reset_metrics()
        
        # FPS calculation
        fps_window = deque(maxlen=30)
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_idx += 1
                
                # Process frame
                frame_start = time.perf_counter()
                result = self.process_frame(frame, frame_idx)
                frame_end = time.perf_counter()
                
                # Calculate FPS
                frame_time = frame_end - frame_start
                if frame_time > 0:
                    fps_window.append(1.0 / frame_time)
                
                current_fps = np.mean(fps_window) if fps_window else 0
                
                # Draw tracking
                display_frame = self.draw_tracking(frame, result, frame_idx)
                
                # Display progress
                if frame_idx % 10 == 0:
                    print(f"[FRAME {frame_idx:4d}] FPS: {current_fps:5.2f} | "
                          f"People: {result['num_people']} | "
                          f"IDs: {result['track_ids']}")
                
                # Display frame
                if display:
                    cv2.imshow('User Tracking Test', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Pause on spacebar
                        cv2.waitKey(0)
                
                # Save frame
                if writer:
                    writer.write(display_frame)
                
                # Check max frames
                if max_frames and frame_idx >= max_frames:
                    break
                    
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_statistics(elapsed_time)
        
        return stats
    
    def _calculate_statistics(self, elapsed_time: float) -> dict:
        """Calculate tracking statistics"""
        m = self.metrics
        
        stats = {
            'total_frames': m['total_frames'],
            'frames_with_detections': m['frames_with_detections'],
            'detection_rate': m['frames_with_detections'] / m['total_frames'] if m['total_frames'] > 0 else 0,
            'total_detections': m['total_detections'],
            'unique_track_ids': len(m['unique_track_ids']),
            'avg_people_per_frame': np.mean(m['people_per_frame']) if m['people_per_frame'] else 0,
            'max_people_per_frame': max(m['people_per_frame']) if m['people_per_frame'] else 0,
            'new_tracks': m['new_tracks'],
            'lost_tracks': m['lost_tracks'],
            'elapsed_time': elapsed_time,
            'processing_fps': m['total_frames'] / elapsed_time if elapsed_time > 0 else 0,
        }
        
        # Track duration statistics
        track_durations = []
        for track_id, frames in m['track_id_history'].items():
            duration = len(frames)
            track_durations.append({
                'track_id': track_id,
                'duration': duration,
                'first_frame': min(frames),
                'last_frame': max(frames)
            })
        
        stats['track_durations'] = sorted(track_durations, key=lambda x: x['duration'], reverse=True)
        
        return stats
    
    def print_report(self, stats: dict):
        """Print detailed tracking report"""
        print("\n" + "="*80)
        print("USER TRACKING TEST REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Total Frames:           {stats['total_frames']}")
        print(f"  Frames with Detections: {stats['frames_with_detections']} ({stats['detection_rate']:.1%})")
        print(f"  Total Detections:       {stats['total_detections']}")
        print(f"  Processing FPS:         {stats['processing_fps']:.2f}")
        print(f"  Elapsed Time:           {stats['elapsed_time']:.2f}s")
        
        print(f"\n👥 TRACKING STATISTICS")
        print(f"  Unique Track IDs:       {stats['unique_track_ids']}")
        print(f"  Avg People per Frame:   {stats['avg_people_per_frame']:.2f}")
        print(f"  Max People per Frame:   {stats['max_people_per_frame']}")
        print(f"  New Tracks Created:     {stats['new_tracks']}")
        print(f"  Tracks Lost:            {stats['lost_tracks']}")
        
        print(f"\n🎯 TRACK DURATION ANALYSIS")
        print(f"  {'Track ID':<10} {'Duration':<10} {'First Frame':<12} {'Last Frame':<12}")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
        
        for track in stats['track_durations'][:10]:  # Show top 10
            print(f"  {track['track_id']:<10} {track['duration']:<10} "
                  f"{track['first_frame']:<12} {track['last_frame']:<12}")
        
        if len(stats['track_durations']) > 10:
            print(f"  ... and {len(stats['track_durations']) - 10} more tracks")
        
        print("\n" + "="*80)
        
        # Performance assessment
        print(f"\n💡 TRACKING ASSESSMENT")
        
        if stats['unique_track_ids'] > 0:
            avg_duration = np.mean([t['duration'] for t in stats['track_durations']])
            print(f"  Average Track Duration: {avg_duration:.1f} frames")
            
            if avg_duration > 50:
                print(f"  ✅ Good tracking consistency (avg duration > 50 frames)")
            elif avg_duration > 20:
                print(f"  ⚠️  Moderate tracking (avg duration 20-50 frames)")
            else:
                print(f"  ❌ Poor tracking (avg duration < 20 frames)")
        
        if stats['detection_rate'] > 0.9:
            print(f"  ✅ Excellent detection rate (>90%)")
        elif stats['detection_rate'] > 0.7:
            print(f"  ⚠️  Good detection rate (70-90%)")
        else:
            print(f"  ❌ Low detection rate (<70%)")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test user tracking on MP4 video')
    parser.add_argument('--video', type=str, required=True, help='Path to MP4 video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display video window')
    parser.add_argument('--save-output', action='store_true',
                       help='Save tracked video to file')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}")
        return
    
    # Create tester
    tester = UserTrackingTester(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    # Run test
    stats = tester.test_video(
        video_path=args.video,
        max_frames=args.max_frames,
        display=not args.no_display,
        save_output=args.save_output
    )
    
    # Print report
    tester.print_report(stats)


if __name__ == '__main__':
    main()
