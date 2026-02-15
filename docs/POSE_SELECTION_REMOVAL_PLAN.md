# Plan: Remove Target Pose Selection Dialog

## Executive Summary
**Objective:** Eliminate the target pose selection dialog to make the system perform true pose recognition rather than pose verification, addressing concerns that pre-selecting the expected pose "assists" the model too much.

**Current Flow:** User selects pose → System expects specific pose → Compares detected vs expected → Feedback  
**New Flow:** User performs any pose → System recognizes pose → Confidence-based feedback

---

## Current System Analysis

### What Uses Target Pose Selection:
1. **Configuration Screen** ([app/app.py#L358-L377](app/app.py#L358-L377))
   - "Target Move" dropdown with 12 pose options
   - Stored in `user_configs[i]['form']`

2. **Feedback Logic** ([app/app.py#L560-L669](app/app.py#L560-L669))
   - Compares `predicted_class` vs `expected_class`
   - Sets `is_correct = (predicted == expected)`
   - Shows "PERFECT"/"GOOD" if match, "ADJUST" if wrong
   - Displays "Goal: {target_pose}" hints

3. **Database Sessions** ([app/database/db_manager.py#L39-L46](app/database/db_manager.py#L39-L46))
   - `sessions` table has `target_pose TEXT` column
   - Saved via `db.start_session(user_id, target_pose=config['form'].get())`

4. **Results Display** ([app/gui/results_window.py#L173-L211](app/gui/results_window.py#L173-L211))
   - Shows "Target Pose" column in session history
   - Uses for filtering/analysis

5. **Form Feedback Generation** ([app/app.py#L674-L740](app/app.py#L674-L740))
   - `generate_form_feedback(target_class, ...)` compares angles against expected pose template
   - Gives specific biomechanical corrections

---

## Proposed Changes

### Phase 1: UI/UX Changes

#### A. Remove Target Pose Selection
**File:** `app/app.py`

**Changes:**
1. **Remove from Config Screen** (Lines 358-377)
   ```python
   # DELETE THIS SECTION:
   ctk.CTkLabel(card, text="Target Move", ...)
   ctk.CTkOptionMenu(card, variable=self.user_configs[i]['form'], values=[...], ...)
   ```

2. **Remove from user_configs** (Line 321)
   ```python
   # BEFORE:
   self.user_configs.append({
       'user': None,
       'viewpoint': ctk.StringVar(value="Front"),
       'form': ctk.StringVar(value="Left Temple Block"),  # DELETE THIS LINE
       'session_id': None
   })
   
   # AFTER:
   self.user_configs.append({
       'user': None,
       'viewpoint': ctk.StringVar(value="Front"),
       'session_id': None
   })
   ```

3. **Simplify Config Card Layout**
   - Make cards shorter (height: 400 → 300)
   - Center remaining elements (user selection + viewpoint)
   - Better visual balance with less clutter

#### B. Update User Experience Flow
**New Experience:**
- User selects only: **Name** + **Viewpoint** (Front/Left Side/Right Side)
- Countdown: "Strike any pose in 3... 2... 1..."
- Snapshot: System recognizes pose automatically
- Feedback: Shows detected pose name + confidence score

---

### Phase 2: Feedback Logic Overhaul

#### A. Remove Verification Logic
**File:** `app/app.py` (Lines 560-669)

**DELETE:**
```python
# Remove target pose matching
target_pose = config['form'].get()
expected_class = class_name_mapping.get(target_pose)
is_correct = (predicted == expected)
```

**REPLACE WITH:** Pure confidence-based scoring
```python
# New logic: Score based purely on model confidence
if predicted_class != 'N/A' and predicted_class.lower() != 'no technique detected':
    if confidence >= 0.40:
        score_text = "EXCELLENT!"
        score_color = "#2ecc71"  # Green
    elif confidence >= 0.30:
        score_text = "GOOD"
        score_color = "#bfff00"  # Lime
    else:
        score_text = "FAIR"
        score_color = "#f1c40f"  # Yellow
else:
    score_text = "NOT DETECTED"
    score_color = "#e74c3c"  # Red
```

#### B. Simplify Feedback Messages
**Display Information:**
1. **User Name** (top)
2. **Detected Pose Name** (large, center) - e.g., "Left Temple Block"
3. **Confidence Score** (below pose name) - e.g., "EXCELLENT!" or "87%"
4. **Optional: Stick Detection** (icon/indicator)

**Remove:**
- "Goal: X" hints
- "Adjust your form" messages (no expected pose to adjust toward)
- Template-based angle feedback (no target to compare against)

#### C. Alternative: Keep Form Feedback (Optional Enhancement)
If you want to keep biomechanical feedback:
```python
# After pose is recognized, check if form is good
if predicted_class in class_name_mapping.values():
    # Use the DETECTED pose as the "template"
    feedback = self.generate_form_feedback(predicted_class, viewpoint, live_angles, landmarks)
    # Show tips like "Extend arm more", "Straighten back", etc.
```

This would provide coaching without revealing what pose was "expected."

---

### Phase 3: Database Schema Changes

#### A. Update Sessions Table
**File:** `app/database/db_manager.py`

**Option 1: Remove target_pose column (Clean Break)**
```python
# Add migration method:
def migrate_remove_target_pose(self):
    """Remove target_pose from sessions table"""
    cursor = self.conn.cursor()
    
    # SQLite doesn't support DROP COLUMN, so recreate table
    cursor.execute('''
        CREATE TABLE sessions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    cursor.execute('''
        INSERT INTO sessions_new (id, user_id, started_at, ended_at)
        SELECT id, user_id, started_at, ended_at FROM sessions
    ''')
    
    cursor.execute('DROP TABLE sessions')
    cursor.execute('ALTER TABLE sessions_new RENAME TO sessions')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions ON sessions(user_id)')
    self.conn.commit()
```

**Option 2: Keep column but set NULL (Backward Compatible)**
```python
# In start_session():
# BEFORE:
def start_session(self, user_id, target_pose=None):
    cursor.execute('INSERT INTO sessions (user_id, target_pose) VALUES (?, ?)', 
                   (user_id, target_pose))

# AFTER:
def start_session(self, user_id):
    cursor.execute('INSERT INTO sessions (user_id) VALUES (?)', (user_id,))
```

#### B. Update is_correct Logic
**File:** `app/database/db_manager.py`

**In `save_performance()`:**
```python
# BEFORE: is_correct = True/False (compared to expected)
# AFTER: Remove is_correct or repurpose as "high_confidence"

def save_performance(self, session_id, user_id, pose_detected, confidence, 
                     stick_detected=False, high_confidence=None):
    """
    high_confidence: True if confidence >= threshold, False otherwise
    (Optional: can remove this parameter entirely)
    """
    cursor = self.conn.cursor()
    cursor.execute('''
        INSERT INTO performances 
        (session_id, user_id, pose_detected, confidence, stick_detected, is_correct)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, user_id, pose_detected, confidence, stick_detected, high_confidence))
    self.conn.commit()
```

**Recommended:** Keep `is_correct` column but rename to `high_confidence` or `quality_score` in future migration.

---

### Phase 4: Results Window Updates

#### A. Remove Target Pose Column
**File:** `app/gui/results_window.py`

**Changes:**
```python
# Line 173 - Remove "target_pose" from columns
columns = ("session_id", "started", "duration", "attempts", "avg_confidence")

# Line 177 - Remove heading
# sessions_table.heading("target_pose", text="Target Pose")  # DELETE

# Line 185 - Remove column width
# sessions_table.column("target_pose", width=200)  # DELETE

# Line 211 - Remove from data insertion
# target = session['target_pose'] or "N/A"  # DELETE
```

#### B. Add New Metrics (Optional)
**Suggested Additions:**
1. **Most Detected Pose** - Show which pose the user performed most in the session
2. **Average Confidence** - Mean confidence across all attempts
3. **Pose Variety** - Count of unique poses detected
4. **Consistency Score** - How consistent were the poses (did they perform same pose repeatedly)

```python
# Example new column:
def calculate_session_stats(session_id):
    perfs = self.db.get_session_performances(session_id)
    
    pose_counts = {}
    confidences = []
    for p in perfs:
        pose = p['pose_detected']
        pose_counts[pose] = pose_counts.get(pose, 0) + 1
        confidences.append(p['confidence'])
    
    most_common = max(pose_counts, key=pose_counts.get)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'most_detected_pose': most_common,
        'avg_confidence': f"{avg_conf:.2%}",
        'pose_variety': len(pose_counts)
    }
```

---

### Phase 5: Additional Considerations

#### A. Training/Instruction Mode
**Problem:** Without target pose, how do users know what to practice?

**Solutions:**

1. **Free Practice Mode** (Current plan)
   - Users perform any pose
   - System tells them what it detected
   - Good for testing/exploration

2. **Add Optional Training Mode** (Future enhancement)
   - Two modes: "Free Practice" vs "Guided Training"
   - In Guided Mode: System randomly selects pose and shows silhouette/demo
   - User tries to match it (target not sent to model, only shown to user)

3. **Random Challenge Mode**
   - System displays pose name AFTER countdown
   - User has 3 seconds to execute
   - Tests reaction + technique

#### B. Evaluation Metrics Change
**Impact:** Without expected pose, how do we measure "accuracy"?

**New Metrics:**
1. **Detection Rate** - % of attempts where a valid pose was recognized (confidence > threshold)
2. **Confidence Score** - Average confidence of detected poses
3. **Consistency** - How reliably user can trigger same pose detection
4. **Diversity** - Range of poses successfully performed

**Update Reports:**
```python
# Replace "accuracy = correct/total"
# With "detection_rate = detected/total"
detection_rate = len([p for p in perfs if p['confidence'] > 0.35]) / len(perfs)
avg_confidence = sum(p['confidence'] for p in perfs) / len(perfs)
```

#### C. Model Performance Validation
**Concern:** How do we know if model is working correctly without ground truth?

**Solutions:**
1. **Confusion Matrix Analysis** - During testing, researcher manually labels poses
2. **Confidence Calibration** - Monitor if confidence scores align with visual observations
3. **Stick Detection Correlation** - High confidence should correlate with stick detected
4. **User Feedback** - Add "Report Error" button if user disagrees with detection

---

## Implementation Roadmap

### Step 1: Preparation (1 hour)
- [ ] Backup current database
- [ ] Backup current codebase
- [ ] Create test environment

### Step 2: Core Changes (2-3 hours)
- [ ] Remove target pose dropdown from UI
- [ ] Update `user_configs` initialization
- [ ] Rewrite feedback logic (remove is_correct comparison)
- [ ] Update `start_session()` to not require target_pose
- [ ] Update `save_performance()` calls

### Step 3: Database Migration (1 hour)
- [ ] Write migration script
- [ ] Test on copy of production database
- [ ] Update `db_manager.py` schema

### Step 4: Results Window (1 hour)
- [ ] Remove target_pose column
- [ ] Add new metrics (avg confidence, detection rate)
- [ ] Update queries

### Step 5: Testing (2-3 hours)
- [ ] Test all 13 poses without pre-selection
- [ ] Verify confidence scores are reasonable
- [ ] Check database saves correctly
- [ ] Test multi-user mode
- [ ] Verify results window displays correctly

### Step 6: Documentation (1 hour)
- [ ] Update user manual
- [ ] Update system documentation
- [ ] Add comments explaining new logic

**Total Estimated Time:** 8-10 hours

---

## Code Changes Summary

| File | Lines to Change | Type | Complexity |
|------|----------------|------|------------|
| `app/app.py` | 321, 358-377, 419, 560-669 | Remove/Rewrite | High |
| `app/database/db_manager.py` | 39-46, 110-130 | Schema + Logic | Medium |
| `app/gui/results_window.py` | 173-211 | Remove columns | Low |

**Total:** ~150 lines removed, ~80 lines added/modified

---

## Risk Assessment

### Low Risk:
- ✅ Removing UI elements (clean, no side effects)
- ✅ Changing feedback text (cosmetic)

### Medium Risk:
- ⚠️ Database schema changes (test thoroughly, provide rollback)
- ⚠️ Feedback logic rewrite (ensure confidence thresholds work)

### High Risk:
- ❌ None identified

### Mitigation:
1. Keep old code in comments initially
2. Feature flag: `ENABLE_TARGET_POSE_SELECTION = False` for easy rollback
3. Comprehensive testing with all 13 poses
4. Monitor first week for issues

---

## Expected Benefits

1. **More Challenging Recognition Task**
   - Model must truly recognize poses without hints
   - Better demonstrates model capability

2. **Cleaner User Experience**
   - Fewer configuration steps
   - Faster setup (no pose selection)
   - More spontaneous interaction

3. **Real-World Scenario**
   - Mimics actual usage where system doesn't know what's coming
   - Better for demonstrations/exhibitions

4. **Research Validity**
   - Addresses adviser's concern about "assisting the model"
   - More academically defensible evaluation

5. **Simplified Codebase**
   - ~150 lines removed
   - Less complex feedback logic
   - Easier to maintain

---

## Alternative Approaches

### Option A: Hybrid Mode (Recommended)
Add a settings toggle:
```python
PURE_RECOGNITION_MODE = True  # vs. VERIFICATION_MODE = False
```
- Recognition Mode: No target pose, confidence-based feedback
- Verification Mode: Select target, get correctness feedback
- Let user/instructor choose based on use case

### Option B: Post-Exercise Reveal
- User performs pose without pre-selection
- System makes prediction
- THEN ask user: "What pose were you attempting?"
- Compare and show if system was correct
- Maintains model challenge while providing ground truth for evaluation

### Option C: Silent Target
- System internally tracks a "secret" target (randomly assigned)
- User doesn't see it
- Model still doesn't know (same recognition task)
- Post-session show "You performed X, system expected Y" for evaluation
- Best of both worlds but more complex UX

---

## Recommendation

**Implement: Pure Recognition Mode (No Target Selection)**

**Rationale:**
1. Fully addresses adviser's concern
2. Simplest implementation
3. Most honest demonstration of model capability
4. Better user experience (less setup)
5. Can always add modes later if needed

**Quick Win:** Start with removing target selection UI, see how it feels in practice, then decide on database schema changes.

---

## Questions for Decision

1. **Keep is_correct column?**
   - Option A: Remove entirely (clean break)
   - Option B: Repurpose as `high_confidence` flag
   - **Recommendation:** Repurpose to avoid schema migration

2. **Show detected pose name to user?**
   - Option A: Yes, show in feedback (transparent)
   - Option B: No, just show confidence (mysterious)
   - **Recommendation:** Yes, educational value

3. **Form feedback (angle corrections)?**
   - Option A: Remove entirely (no target to compare)
   - Option B: Keep, use detected pose as template
   - **Recommendation:** Keep with detected pose

4. **Add training mode later?**
   - Option A: Yes, separate mode for guided practice
   - Option B: No, keep it pure recognition only
   - **Recommendation:** Evaluate after testing current changes

---

## Next Steps

1. **Review this plan** with adviser/team
2. **Get approval** on approach
3. **Create feature branch**: `feature/remove-pose-selection`
4. **Implement Phase 1-2** (UI + Logic)
5. **Test extensively**
6. **Decide on database migration** (Phase 3)
7. **Complete implementation**
8. **Deploy and monitor**

---

*Plan created: February 15, 2026*  
*Status: PENDING REVIEW*
