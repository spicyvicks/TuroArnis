# Project Roadmap: TuroArnis

## Overview

TuroArnis is a real-time Arnis (Filipino martial arts) form correction system using computer vision and machine learning. This roadmap tracks the development and deployment phases.

---

## Completed Phases

### Phase 1: Core CV Pipeline
**Status:** Completed
**Goal:** Implement pose detection and analysis using YOLOv8 + MediaPipe
**Deliverables:**
- Person detection with YOLOv8n
- Pose extraction with MediaPipe (33 keypoints)
- Real-time frame processing
- Stick detection with custom YOLO model

### Phase 2: GCN Model Integration
**Status:** Completed
**Goal:** Integrate Graph Convolutional Network for technique classification
**Deliverables:**
- HybridGCN model architecture (35 nodes: 33 body + 2 stick)
- 3 specialist models (front, left, right viewpoints)
- 12 Arnis technique classification
- Confidence thresholding per viewpoint

### Phase 3: Feedback System
**Status:** Completed
**Goal:** Real-time form correction and feedback generation
**Deliverables:**
- Hybrid correction system (GCN features + joint angles)
- Body visibility gating
- Posture analysis
- Prioritized message system

### Phase 4: GUI and User Interface
**Status:** Completed
**Goal:** Multi-mode user interface with CustomTkinter
**Deliverables:**
- Kiosk mode (fullscreen, multi-user)
- Desktop mode (windowed with controls)
- Results/statistics window
- User management dialog
- Animated lesson instructions

### Phase 5: Database and Persistence
**Status:** Completed
**Goal:** SQLite-based session and performance tracking
**Deliverables:**
- User profiles table
- Session tracking
- Performance records with joint angles
- 7-day statistics aggregation

---

## Current Phase

### Phase 5.5: Classification Algorithm Optimization
**Status:** Ready to Execute
**Goal:** Improve thrust vs block classification accuracy through algorithm fixes

**Description:**
Address classification failures where thrusting techniques are incorrectly classified as blocking and vice versa. Implement 6 locked algorithm improvements: 3D angle calculation for depth discrimination, template STD clamping to tighten loose Gaussians, confidence penalty for missing stick, dynamic adaptive thresholds for similar classes, MediaPipe mode audit to prevent temporal bleed. Algorithm fixes only - no retraining required.

**Decisions:** [D1, D2, D3, D4, D5, D6]
**Plans:** 3 plans
- Plan 01: 3D Angle Calculation (D1)
- Plan 02: Inference Algorithm Improvements (D2, D3, D4)
- Plan 03: MediaPipe Mode Audit (D5)

---

### Phase 5.6: Lesson Mode Similarity Feedback
**Status:** Ready to Execute
**Goal:** Implement similarity-based feedback for lesson mode with A/B testing

**Description:**
Replace binary correct/incorrect feedback in lesson mode with continuous similarity scoring showing users how close their pose is to the target technique. Implement 3 approaches (simple average, variance-weighted, multi-factor), apply psychological +5% buffer (65% actual → 70% display), generate contextual tips, and A/B test to find optimal approach.

**Decisions:** [D-01, D-02, D-03, D-04]
**Plans:** 1 plan
- Plan 01: Similarity Feedback System (REQ-5.6-01 through REQ-5.6-06) - Implement all 3 approaches, A/B rotation, UI integration

---

### Phase 6: Packaging and Deployment
**Status:** Ready to Execute
**Goal:** Create standalone Windows executable with PyInstaller
**Requirements:** [PKG-01, PKG-02, PKG-03, PKG-04, PKG-05, PKG-06, PKG-07]

**Description:**
Package the TuroArnis application into a standalone Windows executable that can be distributed and run without Python installation. Handle complex ML library bundling (PyTorch, MediaPipe, Ultralytics), asset inclusion (models, GIFs, images), and create distribution packaging with documentation.

**Decisions:** [D-02, D-04, D-05, D-06]
**Plans:** 3 plans
- Plan 01: Build System Validation (PKG-01, PKG-02, PKG-03, PKG-04) - Validate spec, resource paths, execute clean build
- Plan 02: Executable Testing (PKG-05) - Automated and manual functional testing of packaged app
- Plan 03: Distribution Packaging (PKG-06, PKG-07) - Create ZIP distribution, deployment docs, optional installer

---

## Future Phases

### Phase 7: Distribution and Updates
**Status:** Future
**Goal:** Automated update system and distribution channel

### Phase 8: Performance Optimization
**Status:** Future
**Goal:** GPU acceleration and inference optimization

---

*Roadmap generated: 2026-04-13*
