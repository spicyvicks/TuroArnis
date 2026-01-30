# Repository Reorganization Plan

## Objective
Separate application and training/ML files into distinct folders for better organization.

## New Structure

```
TuroArnis/
├── 📁 app/                           # Application code
│   ├── main_app.py
│   ├── main_image.py
│   ├── gui/
│   ├── computer_vision/
│   ├── database/
│   ├── utils/
│   ├── assets/
│   └── turoarnis.db
│
├── 📁 ml/                            # ML & Training code
│   ├── training/
│   ├── models/
│   ├── experiments/
│   ├── runs/
│   ├── arnis_poses_angles.csv
│   ├── arnis_poses_coordinates.csv
│   ├── yolov8n-pose.pt
│   └── yolov8n.pt
│
├── 📁 dataset/                       # UNCHANGED - stays in root
├── 📁 dataset_aug/                   # UNCHANGED - stays in root
├── 📁 incorrect/                     # UNCHANGED - stays in root
├── 📁 tools/                         # UNCHANGED - shared utilities
├── 📁 docs/                          # UNCHANGED - documentation
├── 📁 archive/                       # UNCHANGED - archive
├── 📁 build/, dist/                  # Build artifacts
├── cleanup_comments.py               # Root utility
├── sort.py                           # Root utility
├── verify_fix.py                     # Root utility
├── requirements.txt                  # UNCHANGED
├── TuroArnis.spec                    # UNCHANGED (will update paths)
├── TuroArnis_ImageTest.spec          # UNCHANGED (will update paths)
└── BUILD*.md, *.md files             # UNCHANGED
```

## Implementation Steps

### 1. Create New Directories
- Create `app/` folder
- Create `ml/` folder

### 2. Move Application Files
```
main_app.py → app/main_app.py
main_image.py → app/main_image.py
gui/ → app/gui/
computer_vision/ → app/computer_vision/
database/ → app/database/
utils/ → app/utils/
assets/ → app/assets/
turoarnis.db → app/turoarnis.db
```

### 3. Move ML/Training Files
```
training/ → ml/training/
models/ → ml/models/
experiments/ → ml/experiments/
runs/ → ml/runs/
arnis_poses_angles.csv → ml/arnis_poses_angles.csv
arnis_poses_coordinates.csv → ml/arnis_poses_coordinates.csv
yolov8n-pose.pt → ml/yolov8n-pose.pt
yolov8n.pt → ml/yolov8n.pt
```

### 4. Update Import Paths
Files that need import updates:
- `app/main_app.py` - imports from gui/, computer_vision/, database/, utils/
- `app/main_image.py` - imports from computer_vision/, database/, utils/
- `app/gui/*.py` - may import from computer_vision/, database/
- `app/computer_vision/*.py` - imports from utils/
- All files in `ml/training/` - imports between training modules

### 5. Update Path References
- Update `TuroArnis.spec` - paths to main_app.py, assets, data files
- Update `TuroArnis_ImageTest.spec` - paths to main_image.py
- Update any hardcoded paths in Python files (resource_path.py usage)
- Update BUILD.md and other documentation

### 6. Update .gitignore (if needed)
- Ensure build artifacts, __pycache__, etc. are ignored in new structure

## GitHub Impact
✅ **NO COMPLICATIONS** - This is just file reorganization
- Git tracks file moves automatically
- Use `git mv` for better move tracking
- Commit message: "Restructure: Separate app and ML files"
- No impact on cloning, pulling, or pushing

## Testing After Reorganization
1. Run main_app.py from new location
2. Run main_image.py from new location
3. Verify PyInstaller builds still work
4. Test training scripts from new location
5. Verify database access works
6. Verify model loading works

## Rollback Plan
If issues arise, we can:
1. Use `git reset --hard` to undo changes
2. Or manually move files back to original locations
