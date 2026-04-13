# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# PyInstaller Specification for TuroArnis
# =============================================================================
# Purpose: Build a standalone Windows executable that bundles the TuroArnis
#          Python application with all ML models, assets, and dependencies.
#
# Target: app/app.py (Kiosk mode - fullscreen, multi-user)
# Output: dist/TuroArnis/TuroArnis.exe with _internal/ folder
#
# Key Features:
# - One-directory mode (D-02) for reliable asset access
# - Console enabled (D-04) for debugging crash output
# - UPX compression enabled (D-05) to reduce size
# - Comprehensive hidden imports for ML libraries
# - torch_geometric workaround for JIT compilation
# =============================================================================

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all
import os
import sys

# DEBUG: Print current working directory
print(f"Current Working Directory: {os.getcwd()}")

# Define paths
project_root = os.path.abspath('.')
app_dir = os.path.join(project_root, 'app')

print(f"Project Root: {project_root}")
print(f"App Dir: {app_dir}")

# =============================================================================
# HIDDEN IMPORTS
# =============================================================================
# These modules are imported dynamically or via C-extensions and must be
# explicitly declared for PyInstaller to include them in the bundle.
#
# ML/CV Libraries:
# - sklearn: Required for preprocessing and model utilities
# - ultralytics: YOLOv8 object detection
# - mediapipe: 33-keypoint pose detection
# - scipy: Scientific computing functions
# - torch_scatter/sparse: PyTorch Geometric dependencies
#
# Data Processing:
# - pandas: Data manipulation (used by some ML libs)
# - networkx: Graph operations (used by torch_geometric)
# - jinja2: Template rendering (used by pandas/pyinstaller)
#
# GUI Frameworks:
# - PIL/Pillow: Image processing for GUI assets
# - customtkinter: Modern Tkinter UI framework
#
# Application Modules:
# - app.gui.*: UI components (results, dialogs, toasts, spinner)
# - app.computer_vision.*: CV pipeline components
# - app.models.gcn.*: GCN model architecture and feature extraction
# - app.database.*: SQLite database management
# - app.utils.*: Resource path resolution and device management
# =============================================================================

hiddenimports = [
    # --- Scikit-learn (required for preprocessing) ---
    'sklearn',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
    
    # --- YOLO/Ultralytics (person and object detection) ---
    'ultralytics',
    
    # --- MediaPipe (33-keypoint pose detection) ---
    'mediapipe',
    
    # --- SciPy (scientific computing, statistics) ---
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    
    # --- Data processing libraries ---
    'networkx',   # Graph operations for torch_geometric
    'pandas',     # Data manipulation
    'jinja2',     # Template rendering
    
    # --- Image processing (PIL/Pillow) ---
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    
    # --- GUI framework ---
    'customtkinter',  # Modern Tkinter UI
    
    # --- Application GUI modules ---
    'app.gui.results_window',
    'app.gui.user_dialog',
    'app.gui.toast',
    'app.gui.loading_spinner',
    
    # --- Computer vision pipeline ---
    'app.computer_vision.pose_analyzer',
    'app.computer_vision.feedback_analyzer',
    'app.computer_vision.feedback_mapper',
    'app.computer_vision.gcn_inference',
    
    # --- GCN model components ---
    'app.models.gcn.model_architecture',
    'app.models.gcn.feature_extraction',
    
    # --- Database and utilities ---
    'app.database.db_manager',
    'app.utils.resource_path',
    'utils.device_manager',  # Explicit top-level import support
    'app.utils.device_manager',
]

# =============================================================================
# DATA FILES (Asset Bundling)
# =============================================================================
# Format: ('source_path', 'destination_in_bundle')
# 
# These assets are bundled into the executable and accessed at runtime via
# the resource_path.py utility which handles both dev and packaged modes.
#
# Assets included:
# - UI graphics and icons (app/assets/ including TA.ico)
# - Instructional GIFs for all 12 techniques × 3 viewpoints
# - GCN model weights (.pth files) and configuration
# - YOLO base models (person detection)
# =============================================================================

datas = [
    # --- UI Assets ---
    # Contains: TA.ico (window icon), UI graphics, loading images
    ('app/assets', 'app/assets'),
    
    # --- Instructional GIFs (36 total: 12 techniques × 3 viewpoints) ---
    # Crown, Left Chest, Left Elbow, Left Eye, Left Knee, Left Temple,
    # Right Chest, Right Elbow, Right Eye, Right Knee, Right Temple, Solar Plexus
    ('lesson/front_gif', 'lesson/front_gif'),
    ('lesson/left_gif', 'lesson/left_gif'),
    ('lesson/right_gif', 'lesson/right_gif'),
    
    # --- ML Models and Configuration ---
    # GCN models: hybrid_gcn_v2_front.pth, hybrid_gcn_v2_left.pth, hybrid_gcn_v2_right.pth
    # Config: gcn_model_config.json, feature_templates.json
    # YOLO stick detector: weights/best.pt
    ('app/models', 'app/models'),
    ('app/models/gcn_model_config.json', 'app/models'),
    
    # --- YOLO base models ---
    # yolov8n.pt: Person detection model (root level for easy access)
    ('yolov8n.pt', '.'),
]

# =============================================================================
# BINARY COLLECTION
# =============================================================================
binaries = []

# =============================================================================
# TORCH_GEOMETRIC WORKAROUND
# =============================================================================
# PROBLEM: torch_geometric uses JIT compilation for some operations. When
# packaged with PyInstaller, the JIT compiler cannot find its source files
# because they're not included in the standard analysis.
#
# SOLUTION: Manually add torch_geometric's source directory to datas so the
# JIT compiler can access the source files at runtime.
#
# REFERENCE: This is a known PyInstaller + torch_geometric issue.
# See: https://github.com/pyg-team/pytorch_geometric/issues
# =============================================================================

# Import torch_geometric to get its installation path
import torch_geometric
tg_path = os.path.dirname(torch_geometric.__file__)
print(f"Adding torch_geometric path: {tg_path}")
datas.append((tg_path, 'torch_geometric'))

# =============================================================================
# AUTO-COLLECTION FOR ML LIBRARIES
# =============================================================================
# These packages have complex dependencies that PyInstaller may miss.
# Use collect_all() to bundle all submodules, data files, and binaries.
# Note: torch_geometric excluded from auto-collection - handled manually above.
# =============================================================================

packages = ['torch_scatter', 'torch_sparse', 'ultralytics', 'mediapipe', 'scipy', 'numpy', 'customtkinter']

for package in packages:
    try:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
        print(f"Collected: {package}")
    except Exception as e:
        print(f"Warning: Could not collect all for {package}: {e}")

# =============================================================================
# EXCLUSIONS (Build size optimization and error prevention)
# =============================================================================
# - torch_geometric.distributed: Causes RPC initialization errors at runtime
# - app.eval_app, app.main_app: Not used in kiosk mode (app.py is entry point)
# =============================================================================

# Filter out torch_geometric.distributed to prevent RPC initialization errors
hiddenimports = [h for h in hiddenimports if not h.startswith('torch_geometric.distributed')]

# Remove duplicate hiddenimports for cleaner build
hiddenimports = list(dict.fromkeys(hiddenimports))
print(f"Total hiddenimports: {len(hiddenimports)}")

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================
# Entry point: app/app.py (Kiosk Mode - fullscreen, multi-user application)
# 
# Path configuration:
# - project_root: Allows absolute imports from project root
# - app_dir: Allows imports relative to app/ directory
#
# Excludes:
# - torch_geometric.distributed: Prevents RPC errors
# - app.eval_app: Evaluation interface (separate entry point, not needed)
# - app.main_app: Development GUI (windowed mode, not needed for kiosk)
# =============================================================================

a = Analysis(
    ['app/app.py'],
    pathex=[project_root, app_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch_geometric.distributed', 'app.eval_app', 'app.main_app'],
    noarchive=False,
)

# =============================================================================
# BUILD OUTPUT CONFIGURATION
# =============================================================================
# Mode: One-directory (COLLECT creates _internal/ folder with dependencies)
# Reason: Asset access is more reliable in one-directory mode
#
# Settings (per locked decisions D-02, D-04, D-05):
# - upx=True: Compress binaries to reduce distribution size
# - console=True: Keep console visible for debugging crash output
# - debug=False: Disable verbose bootloader output (production)
# - icon=app/assets/TA.ico: Application icon (D-06)
#
# Note: To create one-file mode, change exclude_binaries=False and remove
# the COLLECT step. However, one-directory is recommended for this app
# due to ML model loading requirements.
# =============================================================================

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Required for one-directory mode
    name='TuroArnis',
    debug=False,            # Production: no verbose bootloader output
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,               # D-05: UPX compression enabled
    console=True,           # D-04: Keep enabled to see crash errors
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None, # No code signing (deferred to Phase 7)
    entitlements_file=None,
    icon='app/assets/TA.ico'  # D-06: Application icon
)

# One-directory collection: creates _internal/ folder with all dependencies
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TuroArnis',
)
