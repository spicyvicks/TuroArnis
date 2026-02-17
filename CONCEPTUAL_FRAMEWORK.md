# Conceptual Framework
## TuroArnis: An AI-Enhanced Learning System for Filipino Martial Arts

**Version**: 1.0  
**Date**: February 2026  
**Domain**: Human-Computer Interaction × Sports Science × Machine Learning

---

## 1. THEORETICAL FOUNDATIONS

### 1.1 Pedagogical Framework

#### **Experiential Learning Cycle (Kolb, 1984)**
TuroArnis operationalizes Kolb's experiential learning model:

```
        Concrete Experience
         (Perform Pose)
              ↓
    ┌─────────────────────┐
    │                     │
Reflective Observation ←──┼──→ Abstract Conceptualization
    │  (View Feedback)    │      (Understand Correction)
    │                     │
    └─────────────────────┘
              ↓
      Active Experimentation
        (Adjust & Retry)
```

**Application**:
- **Concrete Experience**: User physically performs Arnis stance
- **Reflective Observation**: System provides real-time visual feedback
- **Abstract Conceptualization**: Feedback explains *why* pose is incorrect (biomechanical principles)
- **Active Experimentation**: User adjusts based on specific correction guidance

#### **Deliberate Practice Theory (Ericsson et al., 1993)**
The system embodies the four components of deliberate practice:

1. **Well-Defined Task**: Each of the 13 stances has clear success criteria
2. **Immediate Feedback**: <200ms latency from action to feedback
3. **Opportunity for Repetition**: Session-based practice with performance tracking
4. **Error Focus**: System prioritizes critical errors (severity classification)

#### **Zone of Proximal Development (Vygotsky, 1978)**
Adaptive scaffolding through three feedback tiers:
- **Beginner**: "Extend your right elbow" (specific instruction)
- **Intermediate**: "Your right elbow is at 120°, target is 160-180°" (numerical)
- **Advanced**: Visual overlay only (expert-level self-correction)

---

### 1.2 Biomechanical Framework

#### **Kinematic Chain Model**
Arnis stances are analyzed as linked segment models:

```
Root: Pelvis (Hip Center)
├── Lower Body Chain
│   ├── Hip Joint (23, 24)
│   ├── Knee Joint (25, 26)
│   └── Ankle Joint (27, 28)
└── Upper Body Chain
    ├── Shoulder Joint (11, 12)
    ├── Elbow Joint (13, 14)
    ├── Wrist Joint (15, 16)
    └── Weapon Endpoint (33, 34)
```

**Key Insight**: Correct form requires specific angular relationships between segments, not just absolute positions.

#### **Degrees of Freedom Problem**
The human body has ~244 degrees of freedom, but Arnis stances constrain these to functional patterns:

- **Elbow angles**: 160-180° (extension) for thrusts, 80-130° (flexion) for blocks
- **Shoulder angles**: Position weapon at specific targets (temple, chest, knee)
- **Grip angles**: 80-120° (stick-arm relationship)
- **Knee angles**: 150-180° (stable stance)

**System Approach**: Reduces 244 DoF to 6 primary joint angles + 30 geometric relationships.

#### **Motor Control Theory - Schema Theory (Schmidt, 1975)**
Users develop generalized motor programs (GMPs) through:

- **Recall Schema**: "What does a thrust feel like?" (proprioceptive memory)
- **Recognition Schema**: System confirms whether observed pose matches expected outcome
- **Parameterization**: Adjusting force, speed, and range while maintaining form

---

### 1.3 Machine Learning Theory

#### **Representation Learning Hierarchy**

```
Level 1: Pixel Space (640×480×3)
    ↓ YOLO Detection
Level 2: Object Space (Bounding Boxes)
    ↓ MediaPipe Pose
Level 3: Keypoint Space (33×3 coordinates)
    ↓ Feature Engineering
Level 4: Semantic Space (Joint Angles)
    ↓ Graph Construction
Level 5: Graph Space (35 nodes, 30 edges)
    ↓ GCN Processing
Level 6: Classification Space (13 classes)
```

#### **Graph Convolutional Networks for Pose**

**Why Graphs?**
- Human pose is inherently non-Euclidean (no grid structure)
- Joints have irregular connectivity (elbow connects to shoulder and wrist)
- Symmetry properties (left/right correspondence)

**Graph Fourier Transform Intuition**:
Just as CNNs detect spatial patterns in images using convolutions, GCNs detect structural patterns in graphs using spectral convolutions:

```
Traditional CNN:    Image (grid) → Conv2D → Feature Maps
GCN:               Pose (graph) → GCNConv → Node Embeddings
```

**Message Passing Interpretation**:
- Layer 1: Local joint relationships (elbow affects wrist)
- Layer 2: Limb-level coordination (arm affects leg balance)
- Layer 3: Whole-body posture (stance affects weapon position)

#### **Viewpoint-Invariant Recognition**

**Problem**: Same pose looks different from different camera angles.

**Solution**: 
- **Training**: Viewpoint augmentation (rotate camera virtually)
- **Inference**: Specialist models (front/left/right)
- **Future**: Equivariant networks (SE(3) transformations)

---

## 2. CONCEPTUAL SYSTEM ARCHITECTURE

### 2.1 Multi-Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  • Visual Feedback Overlay                                    │
│  • Confidence Visualization                                   │
│  • Session Management Interface                               │
│  • Performance Analytics Dashboard                            │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                  FEEDBACK GENERATION LAYER                   │
│  • Form Correctness Assessment                                │
│  • Error Prioritization (critical > warning > info)          │
│  • Actionable Correction Generation                           │
│  • Temporal State Tracking                                    │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION LAYER                       │
│  • GCN Inference (Graph → Class Probabilities)               │
│  • Viewpoint Selection (Front/Left/Right)                    │
│  • Confidence Thresholding                                    │
│  • Prediction Smoothing                                       │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION LAYER                     │
│  • Node Feature Computation (6 features × 35 nodes)          │
│  • Hybrid Feature Computation (30 similarity scores)         │
│  • Graph Construction (Skeleton Edges)                       │
│  • Temporal Feature Aggregation                              │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                  PERCEPTION LAYER                            │
│  • Person Detection (YOLOv8n + ByteTrack)                    │
│  • Pose Estimation (MediaPipe 33-landmark)                   │
│  • Weapon Detection (YOLOv8-Pose 2-keypoint)                 │
│  • Coordinate Normalization                                  │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│  • Video Stream (Webcam/USB Camera)                          │
│  • Frame Preprocessing (Resize, Color Space)                 │
│  • Frame Buffering (Temporal Consistency)                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Information Flow Model

```
Perception → Representation → Decision → Feedback → Action
    ↓              ↓              ↓          ↓        ↓
 Camera      Graph Structure   Classifier   UI      User
  Pixels      Features          Output      Text    Adjusts
                                        Visual
```

**Data Transformation at Each Stage**:

| Stage | Input | Output | Transformation |
|-------|-------|--------|----------------|
| Input | Light (photons) | Digital Image (640×480×3) | Photoelectric conversion |
| Perception | Image | 33 Keypoints + 2 Stick Points | Neural network inference |
| Feature Extraction | Raw Keypoints | Graph (35×6) + Hybrid (30) | Geometric computation |
| Classification | Graph | Class Distribution (13) | GCN forward pass |
| Feedback Generation | Class + Geometry | Text + Visual Cues | Rule-based analysis |
| Presentation | Feedback Data | UI Elements | Rendering pipeline |

---

## 3. USER EXPERIENCE CONCEPTUAL MODEL

### 3.1 Learning Loop

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   SELECT     │────▶│   OBSERVE    │────▶│   ATTEMPT    │
│  (Choose Form)│     │  (Watch Demo)│     │  (Perform)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│   IMPROVE    │◀────│   ANALYZE    │◀────│   RECEIVE    │
│  (Adjust Form)│     │  (Review Log)│     │  (Feedback)  │
└──────────────┘     └──────────────┘     └──────────────┘
```

**Cognitive Load Management**:
- **Intrinsic**: Learning martial arts movement patterns
- **Extraneous**: System interface complexity (minimized)
- **Germane**: Understanding biomechanical principles (maximized)

### 3.2 Engagement Model

**Core Loop (Short-Term)**:
1. **Challenge**: Achieve correct form
2. **Action**: Physical movement
3. **Feedback**: Immediate visual/text response
4. **Progress**: Confidence score increase

**Meta Loop (Long-Term)**:
1. **Skill Acquisition**: Master individual stances
2. **Session Completion**: Accumulate practice time
3. **Progress Tracking**: View historical improvement
4. **Mastery**: Achieve consistent high confidence

**Motivational Mechanics**:
- **Autonomy**: Choose which form to practice
- **Competence**: Clear progression metrics
- **Relatedness**: (Future: Community features)

---

## 4. FEEDBACK SYSTEM CONCEPTUAL MODEL

### 4.1 Multi-Modal Feedback

```
┌─────────────────────────────────────────────────────┐
│              FEEDBACK CHANNELS                      │
├──────────────┬──────────────┬───────────────────────┤
│   VISUAL     │   NUMERICAL  │     TEXTUAL           │
├──────────────┼──────────────┼───────────────────────┤
│ Skeleton     │ Confidence   │ Specific Corrections  │
│ Overlay      │ Percentage   │ "Extend right elbow"  │
│              │              │                       │
│ Color Coding │ Angle        │ Severity Indicators   │
│ (Green/Red)  │ Measurements │ "Critical Error"      │
│              │              │                       │
│ Bounding Box │ FPS Counter  │ Encouragement         │
│              │              │ "Perfect Form!"       │
└──────────────┴──────────────┴───────────────────────┘
```

### 4.2 Error Taxonomy

**Critical Errors** (Must fix):
- Wrong stance entirely (predicted class ≠ target)
- Weapon grip angle severely incorrect (>30° off)
- Major joint deviation (>40° from target)

**Warnings** (Should fix):
- Minor joint misalignment (15-40° off)
- Posture issues (shoulder/hip alignment)
- Stick not detected

**Suggestions** (Optional improvements):
- Refinement opportunities
- Balance adjustments
- Breathing reminders

### 4.3 Temporal Feedback Strategy

**Immediate** (Frame-level):
- Color-coded skeleton overlay
- Current prediction label

**Short-term** (State-level, ~1-2 seconds):
- Form correctness confirmation
- "Hold this position" prompts

**Session-level** (Minutes):
- Accuracy statistics
- Attempt count
- Improvement trends

---

## 5. KNOWLEDGE REPRESENTATION

### 5.1 Pose Ontology

```
ArnisStance
├── ThrustStance
│   ├── HighThrust
│   │   ├── crown_thrust_correct
│   │   ├── left_eye_thrust_correct
│   │   └── right_eye_thrust_correct
│   ├── MidThrust
│   │   ├── left_chest_thrust_correct
│   │   ├── right_chest_thrust_correct
│   │   └── solar_plexus_thrust_correct
│   └── (LowThrusts - future expansion)
├── BlockStance
│   ├── HighBlock
│   │   ├── left_temple_block_correct
│   │   └── right_temple_block_correct
│   ├── MidBlock
│   │   ├── left_elbow_block_correct
│   │   └── right_elbow_block_correct
│   └── LowBlock
│       ├── left_knee_block_correct
│       └── right_knee_block_correct
└── NeutralStance
    └── neutral_stance
```

### 5.2 Feature Hierarchy

```
Geometric Features
├── Joint Angles (6)
│   ├── Left/Right Elbow
│   ├── Left/Right Shoulder
│   └── Left/Right Knee
├── Heights Relative to Hip (6)
│   ├── Wrists, Elbows
│   └── Stick Tip/Grip
├── Horizontal Positions (4)
│   ├── Wrists
│   └── Stick
├── Weapon Orientation (3)
│   ├── Stick Angle
│   ├── Direction Vector
│   └── Grip Angle
└── Expert Features (11)
    ├── Tip vs Body Landmarks
    ├── Hand vs Body Landmarks
    └── Stance Metrics
```

---

## 6. EVALUATION FRAMEWORK

### 6.1 Learning Effectiveness Metrics

**Performance Metrics**:
- Classification Accuracy (system's correctness)
- Confidence Score (system's certainty)
- Form Correctness Rate (user's correctness)

**Learning Metrics**:
- Time to Mastery (sessions until consistent >80%)
- Error Reduction Rate (improvement curve slope)
- Retention Rate (performance after time gap)

**Engagement Metrics**:
- Session Duration
- Practice Frequency
- Form Diversity (variety of stances practiced)

### 6.2 System Performance Metrics

**Latency Budget** (Total <100ms for real-time feel):
- Camera capture: 33ms (30 FPS input)
- Person detection: 30ms
- Pose estimation: 30ms
- Weapon detection: 60ms (every 4th frame)
- Feature extraction: 5ms
- Classification: 5ms
- Feedback generation: 2ms
- Rendering: 16ms (60 FPS output)

**Accuracy Metrics**:
- Per-class precision/recall
- Confusion matrices
- Viewpoint-specific performance

---

## 7. CONCEPTUAL INNOVATIONS

### 7.1 Hybrid Intelligence

**Human-AI Collaboration Model**:
```
AI Handles:                    Human Handles:
• Precision measurement        • Intention/context
• Tireless repetition          • Motivation/persistence
• Objective assessment         • Qualitative feel
• Pattern recognition          • Creative adaptation
• Consistency                  • Intuition
```

### 7.2 Embodied AI

Unlike virtual assistants or recommendation systems, TuroArnis engages with the user's physical body:
- **Input**: Body position (not text/voice)
- **Processing**: Biomechanical analysis (not semantic)
- **Output**: Physical correction guidance (not information)

### 7.3 Situated Learning

Learning happens in context:
- Authentic environment (home dojo)
- Real weapon (rattan stick)
- Actual physical effort
- Immediate application of feedback

---

## 8. THEORETICAL IMPLICATIONS

### 8.1 For Motor Learning Research

1. **Feedback Frequency**: Tests optimal timing for motor skill correction
2. **Multimodal Feedback**: Explores which feedback channels most effective
3. **Individual Differences**: Adapts to different learning speeds
4. **Retention**: Enables longitudinal study of skill retention

### 8.2 For AI Applications

1. **Domain Adaptation**: How to adapt general pose models to specific domains
2. **Viewpoint Robustness**: Specialist vs. generalist model trade-offs
3. **Explainable AI**: Making classification decisions interpretable
4. **Human-in-the-Loop**: Real-time human-AI interaction patterns

### 8.3 For Cultural Preservation

1. **Digitization of Intangible Heritage**: Encoding martial arts knowledge
2. **Accessibility**: Democratizing access to expert instruction
3. **Standardization**: Creating objective benchmarks for form correctness

---

## 9. CONCEPTUAL LIMITATIONS & FUTURE DIRECTIONS

### 9.1 Current Limitations

**Theoretical**:
- Static poses only (no movement sequences)
- Discrete classification (no continuous quality metrics)
- Single practitioner (no partner drills)

**Practical**:
- Fixed camera viewpoint
- Lighting requirements
- Weapon dependency

### 9.2 Future Conceptual Extensions

**Temporal Modeling**:
```
Current: Frame → Class
Future: Sequence → Movement Pattern → Class + Quality
```

**Multi-Agent**:
```
Current: Single person
Future: Student + Instructor (or sparring partner)
```

**Augmented Reality**:
```
Current: Screen-based feedback
Future: HMD overlay showing 3D ghost instructor
```

**Biometric Integration**:
```
Current: Visual analysis only
Future: EMG sensors + heart rate for effort/tension analysis
```

---

## 10. SUMMARY: CONCEPTUAL MODEL AT A GLANCE

```
┌─────────────────────────────────────────────────────────────┐
│               TUROARNIS CONCEPTUAL CORE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT           PROCESSING           OUTPUT                │
│  ─────           ─────────           ──────                 │
│                                                             │
│  Physical        Graph Neural        Form Correction        │
│  Movement    →   Network        →    Guidance               │
│  (Body)          (AI)                (Multimodal)           │
│                                                             │
│  THEORY:         THEORY:             THEORY:                │
│  • Experiential  • Representation    • Deliberate Practice │
│  • Biomechanics    Learning          • Feedback Loops      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Core Value Proposition**:
> "Transform physical practice into data-informed learning through computer vision and graph-based machine learning, making expert-level form correction accessible anytime, anywhere."

---

**References**:
- Ericsson, K. A., Krampe, R. T., & Tesch-Römer, C. (1993). The role of deliberate practice in the acquisition of expert performance.
- Kolb, D. A. (1984). Experiential learning: Experience as the source of learning and development.
- Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
- Schmidt, R. A. (1975). A schema theory of discrete motor skill learning.
- Vygotsky, L. S. (1978). Mind in society: The development of higher psychological processes.

---

*This conceptual framework provides the theoretical foundation for understanding TuroArnis as both a technical system and a pedagogical tool. It bridges computer science, sports science, and educational psychology.*
