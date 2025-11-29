# Tracking Improvements for Multi-User System

## Current Issue: SORT Tracker Limitations

**Trust Level: ⚠️ Medium-Low**

### Problems with current SORT implementation:
1. **ID Switching**: People crossing paths swap IDs
2. **Lost Tracking**: Brief occlusions cause new IDs
3. **No Re-ID**: Can't recognize users who leave and return
4. **Pure IoU matching**: Only uses bounding box overlap, not appearance

### Recommended Solutions:

## Option 1: DeepSORT (Recommended) ⭐

**Pros:**
- Adds appearance-based re-identification
- Much more robust ID consistency
- Handles occlusions better
- Well-tested and proven

**Implementation:**
```bash
pip install deep-sort-realtime
```

**Code changes:**
```python
from deep_sort_realtime.deepsort_tracker import DeepSort

# In PoseAnalyzer.__init__:
self.tracker = DeepSort(
    max_age=90,           # Keep tracks for 90 frames
    n_init=3,             # Confirm track after 3 detections
    max_iou_distance=0.7, # IoU threshold
    embedder="mobilenet"  # Use MobileNet for appearance features
)

# In process_frame:
detections = []
for box in r.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    # Format: ([left, top, width, height], confidence, class)
    detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

tracked_persons = self.tracker.update_tracks(detections, frame=frame)

# Extract results
for track in tracked_persons:
    if not track.is_confirmed():
        continue
    track_id = track.track_id  # Persistent ID
    bbox = track.to_ltrb()     # [x1, y1, x2, y2]
```

## Option 2: ByteTrack (Fastest) ⚡

**Pros:**
- State-of-the-art accuracy
- Very fast
- Handles low-confidence detections well

**Implementation:**
```bash
pip install bytetrack
```

## Option 3: Improve Current SORT

**Quick fixes without changing library:**

### Fix 1: Increase max_age
```python
# In pose_analyzer.py
self.tracker = Sort(max_age=300, min_hits=3, iou_threshold=0.3)
```
- Keeps IDs alive for 10 seconds instead of 3

### Fix 2: Add Position Constraints
```python
def validate_user_assignment(self, person_id, previous_bbox, current_bbox):
    """Check if person hasn't moved too far (likely same person)"""
    prev_center = ((previous_bbox[0] + previous_bbox[2])/2, 
                   (previous_bbox[1] + previous_bbox[3])/2)
    curr_center = ((current_bbox[0] + current_bbox[2])/2,
                   (current_bbox[1] + current_bbox[3])/2)
    
    distance = np.sqrt((prev_center[0] - curr_center[0])**2 + 
                      (prev_center[1] - curr_center[1])**2)
    
    # If moved more than 200 pixels between frames, likely ID switch
    if distance > 200:
        return False
    return True
```

### Fix 3: Add Manual Re-assignment Button
- Add UI button: "Reassign Users"
- Allow user to manually fix ID swaps
- Pause tracking, show current assignments, let user swap

### Fix 4: Visual Confirmation
- Show colored bounding boxes per user (Person #1 = Red, #2 = Blue, etc.)
- Add "Freeze Assignments" button to lock current ID→User mapping
- Display confidence meter for tracking quality

## Option 4: Face Recognition Fallback

For ultimate accuracy when people have visible faces:

```bash
pip install face-recognition
```

```python
import face_recognition

# During initial assignment, capture face encoding
def capture_user_face_encoding(frame, bbox):
    face_encodings = face_recognition.face_encodings(frame, [bbox])
    if face_encodings:
        return face_encodings[0]
    return None

# Store with user assignment
self.user_face_encodings = {person_id: encoding}

# Re-identify when ID switches suspected
def reidentify_by_face(frame, bbox):
    current_encoding = face_recognition.face_encodings(frame, [bbox])
    if not current_encoding:
        return None
    
    for person_id, stored_encoding in self.user_face_encodings.items():
        distance = face_recognition.face_distance([stored_encoding], current_encoding[0])
        if distance < 0.6:  # Match threshold
            return person_id
    return None
```

## Recommendation:

**Short-term (Quick Fix):**
1. Increase `max_age=300` in SORT
2. Add colored bounding boxes per user
3. Add "Reassign Users" manual button

**Long-term (Robust Solution):**
1. Switch to DeepSORT
2. Add face recognition fallback
3. Implement position validation

## Testing Tracking Quality:

Add this to main_app.py to monitor tracking stability:

```python
def check_tracking_quality(self):
    """Monitor for potential ID switches"""
    if not hasattr(self, 'previous_positions'):
        self.previous_positions = {}
        return
    
    for person_id, result in self.last_known_results.items():
        current_bbox = result['bbox']
        current_center = ((current_bbox[0] + current_bbox[2])/2,
                         (current_bbox[1] + current_bbox[3])/2)
        
        if person_id in self.previous_positions:
            prev_center = self.previous_positions[person_id]
            distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
            
            # Alert if moved > 300px in one frame (likely ID switch)
            if distance > 300:
                print(f"⚠️ WARNING: Person #{person_id} jumped {distance:.0f}px - possible ID switch!")
                self.show_reassignment_alert()
        
        self.previous_positions[person_id] = current_center
```

## Bottom Line:

**Current SORT tracker is OK for:**
- Single user scenarios
- Static practice (minimal movement)
- Controlled environment

**NOT reliable for:**
- Multiple people crossing paths
- People entering/leaving frame
- Close proximity practice
- Long sessions (>5 minutes)

**Recommendation: Implement DeepSORT for production use.**
