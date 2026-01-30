# TuroArnis - Deployment Build Instructions

## Quick Build

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Build executable
pyinstaller TuroArnis.spec --clean

# 3. Find executable
# The packaged app will be in: dist/TuroArnis.exe
```

## Testing the Packaged App

```bash
# Run the executable
.\dist\TuroArnis.exe
```

## What Was Fixed

All critical deployment blockers have been resolved:

✅ **Resource Path Handling**: Created `utils/resource_path.py` with PyInstaller-aware path resolution
✅ **Database Location**: Moved to `%APPDATA%\TuroArnis\turoarnis.db` (persists across updates)
✅ **Model Configuration**: `active_model.json` now uses relative paths
✅ **Icon**: Application icon added from `assets/TA.ico`

## Notes

- **First Run**: Creates database automatically
- **Subsequent Runs**: Uses existing session/performance data
- **Database Path**:
  - Development: `%APPDATA%\TuroArnis\turoarnis.db`
  - Will be created at: `%APPDATA%\TuroArnis\turoarnis.db`
- Persists across app updates
- Can be backed up by users

### Models Bundled
- Base YOLO model: `yolov8n.pt`
- Stick detector: `runs/pose/arnis_stick_detector/weights/best.pt` (if exists)
- Active ensemble model: `v023_ensemble_soft_nongs` and dependencies

### First Run
- App will create database automatically
- User selection dialog appears first
- All 12 arnis forms available immediately

## Troubleshooting

If the build fails:
- Ensure all paths in active_model.json are relative (done)
- Check that icon file exists: `assets/TA.ico`
- Run with `--clean` flag to clear cached build files

## File Size
Expected size: ~500MB-1GB (due to TensorFlow, MediaPipe, and models)
