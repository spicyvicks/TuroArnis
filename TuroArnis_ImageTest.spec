# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for TuroArnis Image Tester
Bundles all dependencies, models, and test image into single executable
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

import os

# Collect MediaPipe data files
datas = collect_data_files('mediapipe')

# Add test image
if os.path.exists('Left Temple Block.jpg'):
    datas.append(('Left Temple Block.jpg', '.'))
if os.path.exists('Right Eye Thrust.jpg'):
    datas.append(('Right Eye Thrust.jpg', '.'))

# Add your models if they exist
if os.path.exists('models/arnis_coordinates_classifier.keras'):
    datas.append(('models/arnis_coordinates_classifier.keras', 'models'))
if os.path.exists('models/label_encoder.joblib'):
    datas.append(('models/label_encoder.joblib', 'models'))

# Add all ensemble model directories
for item in os.listdir('models'):
    item_path = os.path.join('models', item)
    if os.path.isdir(item_path) and item.startswith('v0'):
        datas.append((item_path, f'models/{item}'))

# Add active_model.json
if os.path.exists('models/active_model.json'):
    datas.append(('models/active_model.json', 'models'))

# Add stick detector model if it exists
if os.path.exists('runs/pose/arnis_stick_detector/weights/best.pt'):
    datas.append(('runs/pose/arnis_stick_detector/weights/best.pt', 'runs/pose/arnis_stick_detector/weights'))

# Add YOLO base models
if os.path.exists('yolov8n-pose.pt'):
    datas.append(('yolov8n-pose.pt', '.'))
if os.path.exists('yolov8n.pt'):
    datas.append(('yolov8n.pt', '.'))

# Add assets if folder exists
if os.path.exists('assets'):
    datas.append(('assets', 'assets'))

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

binaries = []

a = Analysis(
    ['main_image.py'],
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
    name='TuroArnis_ImageTest',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/TA.ico',
)
