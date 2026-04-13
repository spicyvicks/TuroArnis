# TuroArnis Build Guide

This guide explains how to build the TuroArnis standalone executable using PyInstaller.

## Prerequisites

### Required Software

1. **Python 3.11** (required for compatibility with PyTorch 2.1.0)
   - Download from [python.org](https://www.python.org/downloads/release/python-3110/)
   - Select "Add Python to PATH" during installation

2. **Windows 10 or 11**
   - 64-bit architecture required
   - 8GB RAM minimum (16GB recommended)

3. **Git** (optional, for cloning)
   - Download from [git-scm.com](https://git-scm.com/)

### Virtual Environment Setup

Create and activate a Python 3.11 virtual environment:

```powershell
# Create virtual environment
python -m venv venv_311

# Activate (PowerShell)
.\venv_311\Scripts\activate

# Verify activation
which python  # Should show path containing venv_311
```

### Dependency Installation

Install all required packages:

```powershell
pip install -r requirements.txt
```

This will install:
- PyInstaller 6.12.0
- PyTorch 2.1.0 (CPU version)
- PyTorch Geometric 2.4.0
- MediaPipe, Ultralytics, OpenCV
- All GUI and utility libraries

**Verification:**
```powershell
python -c "import PyInstaller; print(PyInstaller.__version__)"
# Should output: 6.12.0
```

## Build Process

### Quick Build

Run the build automation script:

```powershell
python scripts/build_app.py
```

This will:
1. Validate all required files exist (models, GIFs, assets)
2. Clean previous build artifacts
3. Run PyInstaller (takes 5-10 minutes)
4. Verify the output executable
5. Display build statistics

### Build with Options

```powershell
# Dry run (validate only, don't build)
python scripts/build_app.py --dry-run

# Force clean build (remove all previous artifacts first)
python scripts/build_app.py --clean

# Skip pre-flight validation
python scripts/build_app.py --skip-checks
```

### Manual Build (Advanced)

If you need more control, run PyInstaller directly:

```powershell
# Clean first
Remove-Item -Recurse -Force build/, dist/ -ErrorAction SilentlyContinue

# Build
pyinstaller TuroArnis.spec --noconfirm --clean
```

### Build Output

After successful build:

```
dist/TuroArnis/
├── TuroArnis.exe          # Main executable (~60KB)
├── _internal/             # All bundled dependencies
│   ├── python.exe         # Bundled Python runtime
│   ├── torch/             # PyTorch libraries
│   ├── mediapipe/         # MediaPipe models
│   ├── app/               # Bundled application assets
│   │   ├── assets/        # UI graphics, icons
│   │   └── models/        # GCN weights, YOLO models
│   └── lesson/            # Instructional GIFs
└── ...
```

**Expected build size:** 400-600 MB (depends on bundled ML libraries)

## Testing the Build

### Automated Testing

Run the comprehensive test suite:

```powershell
# Full test suite (fast)
python scripts/test_build.py

# Include smoke test (launches executable for 10 seconds)
python scripts/test_build.py --smoke

# Generate detailed report
python scripts/test_build.py --report
```

Tests verify:
- Executable exists and has correct structure
- All critical files are bundled (models, GIFs, config)
- Critical libraries can be imported
- Resource paths resolve correctly
- Database paths are accessible
- Build size is reasonable

### Manual Testing Checklist

1. **Launch the executable:**
   ```powershell
   .\dist\TuroArnis\TuroArnis.exe
   ```

2. **Watch console output** for the first 30 seconds:
   - ❌ Bad: Red error text, "ModuleNotFoundError"
   - ✅ Good: Window opens, no import errors

3. **Test resource loading:**
   - Select any technique (e.g., "Crown")
   - Verify instructional GIF displays correctly
   - Check TA.ico appears in window title bar

4. **Test GCN model loading:**
   - Start session with "Front" viewpoint
   - Verify no "Model file not found" errors in console
   - Pose classification should work (even without stick detection)

5. **Test database creation:**
   - Check `%APPDATA%/TuroArnis/` folder created
   - Verify `turoarnis.db` file exists there

6. **Test all viewpoints:**
   - Try Front, Left, and Right viewpoints
   - Each should load its corresponding GCN model

## Troubleshooting

### "ModuleNotFoundError" on Launch

**Cause:** PyInstaller missed a hidden import.

**Fix:**
1. Note which module is missing (e.g., `sklearn.utils._something`)
2. Edit `TuroArnis.spec`
3. Add to `hiddenimports` list:
   ```python
   hiddenimports = [
       # ... existing imports ...
       'sklearn.utils._something',  # Add missing module
   ]
   ```
4. Rebuild: `python scripts/build_app.py --clean`

### "Model not found" Errors

**Cause:** Model files not bundled correctly.

**Fix:**
1. Verify models exist in source:
   ```powershell
   Test-Path app/models/hybrid_gcn_v2_front.pth
   Test-Path yolov8n.pt
   ```
2. Check `TuroArnis.spec` datas section includes:
   ```python
   datas = [
       ('app/models', 'app/models'),
       ('yolov8n.pt', '.'),
   ]
   ```
3. Rebuild with `--clean`

### torch_geometric JIT Error

**Cause:** torch_geometric can't find its source files at runtime.

**Symptoms:**
```
RuntimeError: Could not find any valid source files for torch_geometric
```

**Fix:**
The spec file includes a workaround (lines 175-179). Verify it's present:
```python
import torch_geometric
tg_path = os.path.dirname(torch_geometric.__file__)
datas.append((tg_path, 'torch_geometric'))
```

If error persists, try rebuilding with PyInstaller debug mode:
```python
# In TuroArnis.spec, change:
debug=False  # to debug=True
```

### UPX Compression Failures

**Symptoms:** Build completes but executable is very large, or UPX errors appear.

**Fix:**
1. Install UPX (optional):
   ```powershell
   # Download from https://upx.github.io/
   # Place upx.exe in project root or PATH
   ```

2. Or disable UPX temporarily:
   ```python
   # In TuroArnis.spec, change:
   upx=True  # to upx=False
   ```

### Build Size Too Large (> 1GB)

**Cause:** Unnecessary files being bundled.

**Fix:**
1. Check what's being bundled:
   ```powershell
   python scripts/test_build.py --report
   # Check build_test_report.json for largest files
   ```

2. Add exclusions to `TuroArnis.spec`:
   ```python
   excludes=['torch_geometric.distributed', 'unnecessary_module'],
   ```

3. Clean and rebuild:
   ```powershell
   python scripts/clean_build.py --all
   python scripts/build_app.py
   ```

### "File in use" Errors During Clean

**Cause:** Executable is running or antivirus is scanning.

**Fix:**
1. Close any running TuroArnis.exe instances
2. Run clean script with force:
   ```powershell
   python scripts/clean_build.py
   ```
3. If still failing, restart PowerShell and try again

## Distribution

### Creating Release Package

After successful build, create a ZIP for distribution:

```powershell
# Navigate to dist folder
cd dist

# Create ZIP (PowerShell 5.0+)
Compress-Archive -Path TuroArnis -DestinationPath TuroArnis-v1.0.zip

# Or using 7-Zip (if installed)
7z a -tzip TuroArnis-v1.0.zip TuroArnis
```

### What to Distribute

Include in release:
- `TuroArnis/` folder (entire dist/TuroArnis/ directory)
- `README.txt` (brief user instructions)

Do NOT include:
- Source code
- build/ directory
- Model training data
- Development scripts

### End-User System Requirements

Minimum requirements for running the packaged app:

- **OS:** Windows 10 (64-bit) or Windows 11
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 1GB free space
- **Camera:** USB webcam (720p or higher recommended)
- **Python:** NOT required (bundled in executable)

### Installation Instructions for End Users

1. Download `TuroArnis-v1.0.zip`
2. Extract to any folder (e.g., `C:\Program Files\TuroArnis\`)
3. Run `TuroArnis.exe`
4. On first run:
   - Database auto-creates in `%APPDATA%/TuroArnis/`
   - Windows may show SmartScreen warning (click "More info" → "Run anyway")

### Uninstallation

1. Delete the `TuroArnis/` folder
2. Optional: Remove user data:
   ```powershell
   # Remove database and settings
   Remove-Item -Recurse "$env:APPDATA\TuroArnis"
   ```

## Build Scripts Reference

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `scripts/build_app.py` | Main build | `--dry-run`, `--clean`, `--skip-checks` |
| `scripts/clean_build.py` | Clean artifacts | `--all`, `--models`, `--dry-run` |
| `scripts/test_build.py` | Test build | `--smoke`, `--verify-only`, `--report` |

## Architecture Notes

### One-Directory vs One-File Mode

TuroArnis uses **one-directory mode** (D-02 decision):

**Why one-directory?**
- More reliable asset access (models, GIFs)
- Faster startup (no extraction needed)
- Easier debugging (can inspect _internal/ contents)
- Better antivirus compatibility

**Trade-off:** Users must distribute the entire folder, not just a single .exe

### Console Window (D-04 Decision)

The console window is kept enabled in production builds:

**Why keep console?**
- Shows startup errors immediately
- Displays model loading progress
- Critical for debugging user issues
- Can be disabled in `TuroArnis.spec` if desired:
  ```python
  console=False  # Hides console (for final release)
  ```

### torch_geometric Workaround

The build includes special handling for torch_geometric:

```python
# TuroArnis.spec lines 175-179
import torch_geometric
tg_path = os.path.dirname(torch_geometric.__file__)
datas.append((tg_path, 'torch_geometric'))
```

This bundles the library's source files so JIT compilation works correctly. Without this, GCN models fail to load.

## Advanced Configuration

### Customizing the Build

Edit `TuroArnis.spec` to modify:

- **Entry point:** Change `['app/app.py']` to use different main file
- **Icon:** Update `icon='app/assets/TA.ico'` path
- **Console:** Set `console=False` to hide console window
- **UPX:** Set `upx=False` to disable compression

### Build Logging

All build output is saved to `build.log`:

```powershell
# View last 50 lines
tail -50 build.log

# Search for errors
Select-String -Path build.log -Pattern "ERROR"
```

### Test Reports

JSON test reports are saved to `build_test_report.json`:

```powershell
# View report
Get-Content build_test_report.json | ConvertFrom-Json | Format-List
```

## Getting Help

If you encounter build issues:

1. Run `python scripts/test_build.py --report` and check the JSON output
2. Check `build.log` for detailed error messages
3. Verify your virtual environment is activated
4. Ensure all model files exist (run `python scripts/build_app.py --dry-run`)
5. Review the troubleshooting section above

---

**Last Updated:** 2026-04-14  
**Build Version:** PyInstaller 6.12.0  
**Target Platform:** Windows 10/11 x64
