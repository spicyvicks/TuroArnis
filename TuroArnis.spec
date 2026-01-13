# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for TuroArnis Desktop App
Bundles all dependencies, models, and assets into single executable
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files
datas = [
    ('models/*.keras', 'models'),
    ('models/*.joblib', 'models'),
    ('models/*.pt', 'models'),
    ('runs/pose/arnis_stick_detector/weights/*.pt', 'runs/pose/arnis_stick_detector/weights'),
    ('*.pt', '.'),  # YOLO base models
    ('assets/*', 'assets'),  # If you have assets folder
]

# Collect hidden imports
hiddenimports = [
    'mediapipe',
    'cv2',
    'ultralytics',
    'tensorflow',
    'sklearn',
    'joblib',
    'ttkbootstrap',
    'PIL',
    'numpy',
    'pandas',
    'filterpy',
    'scipy',
    'matplotlib',
    'yaml',
] + collect_submodules('mediapipe') + collect_submodules('ultralytics')

# Collect binary files
binaries = []

a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TuroArnis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI only)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add 'icon.ico' if you have one
)
