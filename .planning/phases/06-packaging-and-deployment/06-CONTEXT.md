# Phase 06: Packaging and Deployment - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning
**Source:** Technical analysis of existing spec and codebase

---

<domain>
## Phase Boundary

This phase creates a standalone Windows executable from the TuroArnis Python application using PyInstaller. The deliverable is a self-contained distributable that runs without Python installation.

**In Scope:**
- PyInstaller build configuration refinement
- Asset bundling (models, GIFs, icons)
- Hidden imports for ML libraries
- Path resolution for packaged vs development modes
- Build automation scripts
- Testing of packaged executable

**Out of Scope:**
- Windows installer creation (MSI) — deferred to Phase 7
- Code signing certificates
- Auto-update mechanisms

</domain>

<decisions>
## Implementation Decisions

### Locked Decisions (from existing codebase)
- **Build Tool:** PyInstaller 6.12.0 (D-01)
- **Package Mode:** One-directory with `_internal` folder (D-02)
- **Target Script:** `app/app.py` (Kiosk mode) (D-03)
- **Console:** Keep enabled for debugging crash output (D-04)
- **UPX:** Enabled for compression (D-05)
- **Icon:** `app/assets/TA.ico` (D-06)

### the agent's Discretion
- Build script language (Batch vs PowerShell)
- Build output organization
- Testing checklist format
- Documentation detail level

</decisions>

<canonical_refs>
## Canonical References

**Must Read Before Planning/Implementing:**

### Build Configuration
- `TuroArnis.spec` — PyInstaller specification (primary build config)
- `requirements.txt` — Python dependencies
- `.planning/codebase/STACK.md` — Technology stack details

### Path Resolution
- `app/utils/resource_path.py` — Runtime path utilities

### Model Assets
- `app/models/hybrid_gcn_v2_*.pth` — Three GCN specialist models
- `app/models/weights/best.pt` — YOLO stick detector
- `yolov8n.pt`, `yolov8n-pose.pt` — YOLO base models

### Configuration
- `app/models/gcn_model_config.json` — Model paths and thresholds
- `app/models/gcn/feature_templates.json` — Feature statistics

</canonical_refs>

<specifics>
## Specific Requirements

### Critical Hidden Imports (from TuroArnis.spec)
```python
hiddenimports = [
    'sklearn', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs',
    'ultralytics', 'mediapipe', 'scipy.special.cython_special',
    'networkx', 'pandas', 'jinja2', 'PIL', 'PIL.Image', 'PIL.ImageTk',
    'customtkinter', 
    'app.gui.results_window', 'app.gui.user_dialog', 'app.gui.toast',
    'app.computer_vision.pose_analyzer', 'app.computer_vision.feedback_analyzer',
    'app.models.gcn.model_architecture', 'app.database.db_manager',
    'app.utils.resource_path', 'app.utils.device_manager',
]
```

### Asset Bundling (datas in spec)
- `('app/assets', 'app/assets')` — UI graphics
- `('lesson/front_gif', 'lesson/front_gif')` — Instructional GIFs
- `('lesson/left_gif', 'lesson/left_gif')`
- `('lesson/right_gif', 'lesson/right_gif')`
- `('app/models', 'app/models')` — Model configs and weights
- `('yolov8n.pt', '.')` — YOLO base model

### Exclusions (for build size reduction)
- `torch_geometric.distributed` — Prevents RPC errors
- `app.eval_app`, `app.main_app` — Excluded entry points

### torch_geometric Workaround
Manual path collection required:
```python
import torch_geometric
tg_path = os.path.dirname(torch_geometric.__file__)
datas.append((tg_path, 'torch_geometric'))
```

</specifics>

<deferred>
## Deferred Ideas

- Windows installer (.msi) creation — moved to Phase 7
- Code signing for executable trust
- Build artifact compression (.zip with compression)
- One-file mode (currently using one-directory for asset access)

</deferred>

---

*Phase: 06-packaging-and-deployment*
