# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os

# Define the app directory explicitly
app_dir = os.path.abspath('app')

hiddenimports = [
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
    'ultralytics',
    'mediapipe',
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    'gui.results_window',
    'database.db_manager',
    'computer_vision.pose_analyzer',
    'utils.resource_path'
]
hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('mediapipe')

# Collect data files
datas = [
    ('app/assets', 'app/assets'),
    ('app/models', 'app/models'),
    ('deployment_package', 'deployment_package'), # Needed for app.py line 75
    ('yolov8n.pt', '.'),
]
datas += collect_data_files('ultralytics')
datas += collect_data_files('mediapipe')

a = Analysis(
    ['app/app.py'],
    pathex=[app_dir], # Add app/ to path so 'import gui' works
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TuroArnis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Keep True for debugging initial crashes
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
