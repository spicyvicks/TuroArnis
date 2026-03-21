# -*- mode: python ; coding: utf-8 -*-
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

hiddenimports = [
    'sklearn',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
    'ultralytics',
    'mediapipe',
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    'networkx',
    'pandas',
    'jinja2',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'customtkinter', 
    'app.gui.results_window',
    'app.gui.user_dialog',
    'app.gui.toast',
    'app.gui.loading_spinner',
    'app.computer_vision.pose_analyzer',
    'app.computer_vision.feedback_analyzer',
    'app.computer_vision.feedback_mapper',
    'app.computer_vision.gcn_inference',
    'app.models.gcn.model_architecture',
    'app.models.gcn.feature_extraction',
    'app.database.db_manager',
    'app.utils.resource_path',
    'utils.device_manager', # Explicitly add as top-level if code imports it that way
    'app.utils.device_manager',
]

# Explicitly collect data files for critical libraries
datas = [
    ('app/assets', 'app/assets'),
    ('app/assets/lesson_images', 'app/assets/lesson_images'),
    ('app/models', 'app/models'),
    ('app/models/gcn_model_config.json', 'app/models'),
    # Copy deployment_package if it exists, as main_app might use it (though seemingly not directly?)
    # app/main_app.py uses 'app/models/weights/best.pt', not deployment_package
    # But just in case, we include it if present
    ('deployment_package', 'deployment_package'), 
    ('yolov8n.pt', '.'),
]

# Collect robustly
binaries = []

# Packages requiring full collection (Removed torch_geometric to avoid distributed rpc error on build)
packages = ['torch_scatter', 'torch_sparse', 'ultralytics', 'mediapipe', 'scipy', 'numpy', 'customtkinter']

# Manually add torch_geometric source files to datas to fix JIT error
import torch_geometric
tg_path = os.path.dirname(torch_geometric.__file__)
datas.append((tg_path, 'torch_geometric'))

for package in packages:
    try:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception as e:
        print(f"Warning: Could not collect all for {package}: {e}")

# Filter out torch_geometric.distributed from hiddenimports to prevent build errors
hiddenimports = [h for h in hiddenimports if not h.startswith('torch_geometric.distributed')]

# Add python source files from app to ensuring direct imports work if needed
# (Analysis usually handles this, but explicit is safe for data files)

a = Analysis(
    ['app/app.py'],  # TARGET: app.py (Kiosk Mode)
    pathex=[project_root, app_dir], # PATH: Project root AND app dir
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch_geometric.distributed'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TuroArnis',
    debug=False, # Set to True for verbose console output during startup
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Keep Console TRUE to see crash errors
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app/assets/TA.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TuroArnis',
)
