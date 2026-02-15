# User Flow Comparison: Current vs. Proposed

## Current Flow (With Target Pose Selection)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SPLASH SCREEN                                                │
│    "TuroArnis - Arnis Form Correction System"                   │
│    [Press any key to start]                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. USER COUNT SELECTION                                         │
│    "How many users?"                                            │
│    [1 User] [2 Users] [3 Users]                                │
│    [← BACK]                                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. CONFIGURATION SCREEN (Per User)                              │
│    ┌─────────────┬─────────────┬─────────────┐                │
│    │   User 1    │   User 2    │   User 3    │                │
│    ├─────────────┼─────────────┼─────────────┤                │
│    │ [Guest 1▼]  │ [Guest 2▼]  │ [Guest 3▼]  │ ← User Select  │
│    │ Tap to      │ Tap to      │ Tap to      │                │
│    │ change user │ change user │ change user │                │
│    │             │             │             │                │
│    │ Viewpoint   │ Viewpoint   │ Viewpoint   │                │
│    │ [Front|Rt|Lt]│ [Front|Rt|Lt]│[Front|Rt|Lt]│               │
│    │             │             │             │                │
│    │ Target Move │ Target Move │ Target Move │ ← POSE SELECT  │
│    │[Left Temple │[Crown Thrust│[Right Eye   │   (REMOVED)    │
│    │  Block ▼]   │    ▼]       │  Thrust ▼]  │                │
│    └─────────────┴─────────────┴─────────────┘                │
│                                                                 │
│    [← BACK]              [LOCK IN [ENTER]]                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. ZONING VIEW                                                  │
│    Live camera feed divided into zones per user                │
│    "Position yourself in your zone"                             │
│    [✓ CONTINUE [SPACE]]                                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. COUNTDOWN                                                    │
│    Live camera with skeleton overlay                            │
│    "Strike your pose in..."                                     │
│    "3... 2... 1..."                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. SNAPSHOT + ANALYSIS                                          │
│    Frame freezes, analyzing...                                  │
│    "Analyzing your form..."                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. FEEDBACK SCREEN                                              │
│    ┌─────────────┬─────────────┬─────────────┐                │
│    │   User 1    │   User 2    │   User 3    │                │
│    │ "John"      │ "Maria"     │ "Chen"      │                │
│    │             │             │             │                │
│    │  PERFECT!   │    GOOD!    │   ADJUST    │ ← Score        │
│    │   (Green)   │  (Lime)     │   (Red)     │                │
│    │             │             │             │                │
│    │ Perfect     │ Goal: Crown │ Goal: Right │ ← Feedback     │
│    │ form!       │ Thrust      │ Eye Thrust  │   (Compares    │
│    │             │ Extend arm  │ Adjust form │    to target)  │
│    └─────────────┴─────────────┴─────────────┘                │
│                                                                 │
│    [← VIEW RESULTS]         [↻ TRY AGAIN]                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key Issues:**
- ❌ User must pre-select expected pose
- ❌ System compares detected vs. expected (verification, not recognition)
- ❌ Feedback shows "Goal:" hints
- ❌ "Assists" the model by narrowing expectations

---

## Proposed Flow (Pure Recognition Mode)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SPLASH SCREEN                                                │
│    "TuroArnis - Arnis Form Correction System"                   │
│    [Press any key to start]                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. USER COUNT SELECTION                                         │
│    "How many users?"                                            │
│    [1 User] [2 Users] [3 Users]                                │
│    [← BACK]                                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. CONFIGURATION SCREEN (Per User) ✨ SIMPLIFIED                │
│    ┌─────────────┬─────────────┬─────────────┐                │
│    │   User 1    │   User 2    │   User 3    │                │
│    ├─────────────┼─────────────┼─────────────┤                │
│    │ [Guest 1▼]  │ [Guest 2▼]  │ [Guest 3▼]  │ ← User Select  │
│    │ Tap to      │ Tap to      │ Tap to      │                │
│    │ change user │ change user │ change user │                │
│    │             │             │             │                │
│    │ Viewpoint   │ Viewpoint   │ Viewpoint   │                │
│    │ [Front|Rt|Lt]│ [Front|Rt|Lt]│[Front|Rt|Lt]│               │
│    │             │             │             │                │
│    │ ✅ READY    │ ✅ READY    │ ✅ READY    │ ← No pose      │
│    │             │             │             │    selection!  │
│    └─────────────┴─────────────┴─────────────┘                │
│                                                                 │
│    [← BACK]              [LOCK IN [ENTER]]                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. ZONING VIEW                                                  │
│    Live camera feed divided into zones per user                │
│    "Position yourself in your zone"                             │
│    [✓ CONTINUE [SPACE]]                                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. COUNTDOWN                                                    │
│    Live camera with skeleton overlay                            │
│    "Strike any pose in..."  ← Changed from "Strike your pose"  │
│    "3... 2... 1..."                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. SNAPSHOT + ANALYSIS                                          │
│    Frame freezes, analyzing...                                  │
│    "Recognizing your technique..." ← Changed                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. FEEDBACK SCREEN ✨ RECOGNITION-BASED                         │
│    ┌─────────────┬─────────────┬─────────────┐                │
│    │   User 1    │   User 2    │   User 3    │                │
│    │ "John"      │ "Maria"     │ "Chen"      │                │
│    │             │             │             │                │
│    │ Left Temple │ Crown       │ Right Eye   │ ← Detected     │
│    │    Block    │  Thrust     │   Thrust    │    Pose Name   │
│    │             │             │             │                │
│    │ EXCELLENT!  │   GOOD      │    FAIR     │ ← Confidence   │
│    │  (Green)    │  (Lime)     │  (Yellow)   │    Score       │
│    │             │             │             │                │
│    │ 87%         │ 72%         │ 58%         │ ← Optional %   │
│    │ ✓ Stick     │ ✓ Stick     │ ✗ No stick  │ ← Stick status │
│    └─────────────┴─────────────┴─────────────┘                │
│                                                                 │
│    [← VIEW RESULTS]         [↻ TRY AGAIN]                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key Improvements:**
- ✅ No target pose selection required
- ✅ Pure recognition (model identifies pose blindly)
- ✅ Confidence-based feedback (not correctness)
- ✅ Faster setup (fewer clicks)
- ✅ More academically valid

---

## Detailed Step Comparison

| Step | Current Flow | Proposed Flow | Time Saved |
|------|--------------|---------------|------------|
| **1. Splash** | Press any key | Press any key | 0s |
| **2. User Count** | Select 1-3 users | Select 1-3 users | 0s |
| **3. Config** | Select user + viewpoint + **target pose** | Select user + viewpoint only | **~10-15s per user** |
| **4. Zoning** | Position yourself | Position yourself | 0s |
| **5. Countdown** | "Strike your pose" (implied: the one you selected) | "Strike any pose" | 0s |
| **6. Snapshot** | Analyzes for expected pose | Recognizes any pose | 0s |
| **7. Feedback** | Correct/Wrong vs. target | Detected pose + confidence | 0s |

**Total Time Saved:** 10-15 seconds per user (30-45s for 3 users)

---

## User Experience Narratives

### Current Flow (Verification Mode)
> **John's Experience:**
> 1. John presses start
> 2. Selects "1 User"
> 3. Chooses his profile "John Doe"
> 4. Selects "Front" viewpoint
> 5. **Scrolls dropdown, selects "Left Temple Block"** ← Extra step
> 6. Clicks "LOCK IN"
> 7. Positions in zone
> 8. Countdown: "3... 2... 1..."
> 9. Performs Left Temple Block
> 10. System: "PERFECT! ✓" (because it matched expected pose)
> 
> **Issue:** John feels guided. The system "knew" what to look for.

### Proposed Flow (Recognition Mode)
> **John's Experience:**
> 1. John presses start
> 2. Selects "1 User"
> 3. Chooses his profile "John Doe"
> 4. Selects "Front" viewpoint
> 5. ~~Scrolls dropdown, selects pose~~ ← **Removed!**
> 6. Clicks "LOCK IN"
> 7. Positions in zone
> 8. Countdown: "3... 2... 1..."
> 9. Performs Left Temple Block
> 10. System: "Left Temple Block - EXCELLENT! 87% ✓"
> 
> **Benefit:** John is impressed. The system recognized his pose without prompting!

---

## Multi-User Scenario (3 Users)

### Current Flow
```
Time: 0:00 → Select 3 users
Time: 0:05 → User 1: Select name, viewpoint, POSE (10s)
Time: 0:15 → User 2: Select name, viewpoint, POSE (10s)
Time: 0:25 → User 3: Select name, viewpoint, POSE (10s)
Time: 0:35 → Lock in, proceed to zoning
Time: 0:40 → Position in zones
Time: 0:45 → Countdown
Time: 0:48 → Snapshot + Analysis
Time: 0:53 → View feedback (compares to pre-selected poses)
```

### Proposed Flow
```
Time: 0:00 → Select 3 users
Time: 0:05 → User 1: Select name, viewpoint (5s)
Time: 0:10 → User 2: Select name, viewpoint (5s)
Time: 0:15 → User 3: Select name, viewpoint (5s)
Time: 0:20 → Lock in, proceed to zoning (15s faster!)
Time: 0:25 → Position in zones
Time: 0:30 → Countdown: "Strike any pose"
Time: 0:33 → Snapshot + Analysis
Time: 0:38 → View feedback (shows detected poses + confidence)
```

**Setup Time:** 35s → 20s (43% faster)

---

## Session Flow Example

**Scenario:** Training session with instructor

### Current Flow
```
Instructor: "Okay class, today we're practicing Left Temple Block.
            Everyone select 'Left Temple Block' from your dropdown."

[Students configure devices, all selecting same target pose]

Instructor: "Now perform the technique!"

[System compares each student's performance to Left Temple Block]

Result: System shows "CORRECT" or "WRONG" based on match
```

**Problem:** System acts as binary checker, not intelligent recognizer.

### Proposed Flow
```
Instructor: "Okay class, today we're practicing Left Temple Block.
            Just select your names and viewpoint, then perform it."

[Students configure devices - no pose selection]

Instructor: "Now perform the technique!"

[System recognizes whatever pose each student performs]

Result: System shows "Left Temple Block - 87%" or 
        "Left Elbow Block - 62%" (if student did wrong move)
```

**Benefit:** 
- Instructor can see WHO did the right technique
- Students who did wrong move see what system thinks they did
- More realistic assessment of learning

---

## Edge Cases

### Case 1: User Performs Unexpected Pose (Current)
```
Selected: "Left Temple Block"
Performed: "Right Temple Block"
System: "ADJUST - Goal: Left Temple Block" (Red feedback)
```
**Issue:** User confused - they performed perfectly, just different pose.

### Case 1: User Performs Any Pose (Proposed)
```
Selected: Nothing
Performed: "Right Temple Block"
System: "Right Temple Block - EXCELLENT! 94%" (Green feedback)
```
**Benefit:** Honest recognition, no pre-judgment.

---

### Case 2: Model Uncertainty (Current)
```
Selected: "Left Temple Block"
Performed: Ambiguous pose between Left/Right Temple
System: "WRONG - Goal: Left Temple Block" (Red)
```
**Issue:** Harsh feedback, not informative.

### Case 2: Model Uncertainty (Proposed)
```
Selected: Nothing
Performed: Ambiguous pose
System: "Right Temple Block - FAIR (58%)" (Yellow)
```
**Benefit:** Shows uncertainty through confidence score.

---

## Information Architecture

### Current System
```
User Input → Target Pose Selection
            ↓
Model Input → Frame + [No target info actually used by model!]
            ↓
Model Output → Predicted Pose + Confidence
            ↓
App Logic → Compare predicted vs. selected target
            ↓
Feedback → "CORRECT" or "WRONG" + hints
```

### Proposed System
```
User Input → (No target selection)
            ↓
Model Input → Frame only
            ↓
Model Output → Predicted Pose + Confidence
            ↓
App Logic → Evaluate confidence level
            ↓
Feedback → Pose name + confidence rating
```

**Cleaner data flow, fewer steps, no misleading hints.**

---

## Summary

### What Users See
| Aspect | Current | Proposed |
|--------|---------|----------|
| **Config screens** | 3-4 clicks per user | 2-3 clicks per user |
| **Setup time** | ~35s for 3 users | ~20s for 3 users |
| **Countdown prompt** | "Strike your pose" | "Strike any pose" |
| **Feedback text** | "PERFECT/GOOD/ADJUST" + "Goal: X" | "EXCELLENT/GOOD/FAIR" + pose name |
| **Score metric** | Correctness (binary) | Confidence (gradient) |

### What Changes Behind the Scenes
- ❌ Remove target pose dropdown
- ❌ Remove `config['form']` variable
- ❌ Remove expected vs. predicted comparison
- ❌ Remove "Goal:" hints in feedback
- ✅ Add confidence-based scoring
- ✅ Add detected pose name display
- ✅ Show percentage confidence (optional)

### Academic Validity
- **Current:** Pose verification system (less impressive)
- **Proposed:** Pose recognition system (more challenging, defensible)

---

*User Flow Documentation - February 15, 2026*
