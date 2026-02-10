# GCN Environment Setup Guide

## ✅ Environment Status

**Python Version**: 3.12.10  
**Virtual Environment**: `venv/`  
**Status**: ✅ Ready to use

## Installed Packages

### Core Dependencies
- ✅ **PyTorch**: 2.10.0+cpu
- ✅ **PyTorch Geometric**: 2.7.0
- ✅ **torchvision**: 0.25.0+cpu

### Computer Vision
- ✅ **OpenCV**: 4.13.0
- ✅ **MediaPipe**: 0.10.14
- ✅ **Ultralytics (YOLO)**: 8.4.13

### UI & Utilities
- ✅ **CustomTkinter**: 5.2.2
- ✅ **Pillow**: 12.0.0
- ✅ **NumPy**: 2.3.5
- ✅ **joblib**: 1.5.3

## Quick Start

### Activate Virtual Environment

**Windows (PowerShell)**:
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**:
```cmd
.\venv\Scripts\activate.bat
```

### Verify Installation

```bash
python -c "import torch; import torch_geometric; print('✅ Environment ready!')"
```

## Running the GCN FPS Test

Once activated, run the test script:

```bash
# Basic test
python test_gcn_fps.py --video path/to/video.mp4 --viewpoint front

# With display
python test_gcn_fps.py --video path/to/video.mp4 --viewpoint front --display

# Test first 100 frames
python test_gcn_fps.py --video path/to/video.mp4 --viewpoint front --max-frames 100
```

## Environment Details

### Python Location
```
venv\Scripts\python.exe
```

### Site Packages
```
venv\Lib\site-packages\
```

### Installed Package Versions

Run to see all installed packages:
```bash
pip list
```

## Troubleshooting

### Virtual Environment Not Activating

If you get execution policy errors on Windows:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Import Errors

If you encounter import errors, verify all packages are installed:
```bash
pip install -r requirements.txt
```

### PyTorch Geometric Issues

If torch-geometric has issues, install with:
```bash
pip install torch-geometric --no-cache-dir
```

## Next Steps

1. ✅ Environment is set up
2. 📹 Prepare an MP4 video file for testing
3. 🚀 Run the GCN FPS test script
4. 📊 Review performance metrics
5. 🔧 Proceed with GCN integration into main app

## Deactivating

When done, deactivate the virtual environment:
```bash
deactivate
```

---

**Setup Date**: 2026-02-10  
**Python Version**: 3.12.10  
**Platform**: Windows
