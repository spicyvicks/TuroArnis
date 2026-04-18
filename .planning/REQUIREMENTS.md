# Requirements: TuroArnis

## Phase 6: Packaging and Deployment

### PKG-01: Build System Setup
**Priority:** Must Have
**Description:** Configure PyInstaller for creating standalone Windows executable
**Acceptance Criteria:**
- PyInstaller spec file produces working executable
- All dependencies (PyTorch, MediaPipe, Ultralytics, CustomTkinter) are bundled
- UPX compression enabled for smaller file size
- Build completes without errors

### PKG-02: Asset Bundling
**Priority:** Must Have
**Description:** Include all required assets in the packaged application
**Acceptance Criteria:**
- GCN model files (.pth) bundled in correct location
- YOLO model files (.pt) accessible at runtime
- Lesson GIFs (instructional animations) included
- UI assets (icons, images) bundled
- Feature templates JSON accessible

### PKG-03: Hidden Imports Configuration
**Priority:** Must Have
**Description:** Explicitly declare all hidden imports for ML and UI libraries
**Acceptance Criteria:**
- sklearn submodules collected
- torch_scatter and torch_sparse collected
- MediaPipe and Ultralytics fully bundled
- CustomTkinter resources included
- App modules (GUI, CV, models, database) importable

### PKG-04: Runtime Path Resolution
**Priority:** Must Have
**Description:** Ensure resource paths work in both development and packaged modes
**Acceptance Criteria:**
- `_MEIPASS` detection for PyInstaller paths
- `get_resource_path()` utility works in both modes
- Model loading from bundled paths succeeds
- Database initialization in user data directory

### PKG-05: Executable Testing
**Priority:** Must Have
**Description:** Verify the packaged executable runs correctly
**Acceptance Criteria:**
- Application launches without Python installation
- Camera capture works
- Pose detection functional
- GCN inference runs
- GUI displays correctly
- Database operations work

### PKG-06: Build Automation
**Priority:** Should Have
**Description:** Create scripts for reproducible builds
**Acceptance Criteria:**
- Build script cleans previous builds
- Virtual environment activation handled
- PyInstaller runs with correct spec file
- Output organized in dist/ directory
- Build artifacts documented

### PKG-07: Distribution Packaging
**Priority:** Could Have
**Description:** Create distributable package (ZIP or installer)
**Acceptance Criteria:**
- Standalone folder with all required files
- README with installation instructions
- No external dependencies required
- Optional: Windows installer (.msi or .exe)

---

*Requirements version: 1.0*
*Last updated: 2026-04-13*
