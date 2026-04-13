# Codebase Concerns

**Analysis Date:** 2026-04-13

## Critical Issues

### 1. Large Binary Model Files Committed to Git

**Issue:** PyTorch model files (.pt, .pth) totaling ~35+ MB are tracked in the repository.

**Files:**
- `yolov8n.pt` (6.3 MB)
- `yolov8n-pose.pt` (6.5 MB)
- `app/yolov8n.pt` (6.3 MB - duplicate)
- `app/models/hybrid_gcn_v2_*.pth` (multiple, ~5+ MB each)
- `app/models/weights/best.pt` (varies)
- `deployment_package/weights/best.pt`

**Impact:**
- Significantly increases repository size and clone time
- Makes git history heavy and slow
- Binary files cannot be diffed, hindering code review
- Every model update bloats git history permanently

**Recommendation:**
- Add `*.pt` and `*.pth` to `.gitignore`
- Use Git LFS for model files if they must be versioned
- Store models in external storage (S3, cloud bucket) with download scripts

---

### 2. Build Artifacts Committed to Repository

**Issue:** PyInstaller build outputs are present in the repo.

**Directories:**
- `build/` (100+ MB) - Contains `TuroArnis.pkg`, `PYZ-00.pyz` (100MB+), `.toc` files
- `dist/` (60+ MB) - Contains compiled `TuroArnis.exe` and bundled `_internal` directory

**Files:**
- `dist/TuroArnis/TuroArnis.exe` (60MB)
- `dist/TuroArnis/_internal/` (entire Python runtime bundled)
- `build/TuroArnis/*.pkg`, `*.pyz`, `*.toc`

**Impact:**
- Repository size bloated by ~160+ MB of generated artifacts
- These should be regenerated on each build, not versioned
- `dist/` is in `.gitignore` but files were committed before rule was added

**Recommendation:**
- Remove build artifacts from git: `git rm -r build/ dist/`
- Add `build/` and `dist/` to `.gitignore` (already partially done)
- Document build process in README for regeneration

---

### 3. Virtual Environment Committed to Repository

**Issue:** Python virtual environment `venv_311/` is present in the repository.

**Contents:**
- `venv_311/Lib/` (Python packages)
- `venv_311/Scripts/` (executables)
- `venv_311/Include/`

**Impact:**
- Adds unnecessary bloat to repository
- Environment is platform-specific (Windows paths)
- Should be recreated per developer/machine
- Already in `.gitignore` but was committed before rule

**Recommendation:**
- Remove from git: `git rm -r venv_311/`
- Document setup: `python -m venv venv_311` in README

---

### 4. Database Files in Repository

**Issue:** SQLite database files are committed.

**Files:**
- `turoarnis.db` (88 KB)
- `app/turoarnis.db` (64 KB)
- `dist/TuroArnis/turoarnis.db` (36 KB)

**Impact:**
- Database contains runtime data, not source code
- May contain sensitive user data
- Multiple copies suggest sync issues

**Recommendation:**
- Add `*.db` to `.gitignore`
- Add database initialization/migration scripts instead
- Document database schema separately

---

### 5. Test Images and Results in Repository

**Issue:** Large quantities of test images, screenshots, and results are versioned.

**Directories:**
- `trio/` - 13 test images (~3 MB each)
- `results_trio/` - 12 test output images (~2-3 MB each)
- `eval_screenshots/` - front/left/right subdirectories with evaluation images
- `lesson/` - front_gif/, left_gif/, right_gif/ subdirectories
- Root directory images: `ashly_left_chest.jpg`, `indira_crown_thrust.jpg`, `test_result.jpg`, etc.

**Specific large files in root:**
- `ashly_left_chest.jpg` (3.2 MB)
- `indira_crown_thrust.jpg` (3.2 MB)
- `Copy.jpg`, `Copy 2.jpg` (3+ MB each)
- `output_*.jpg` files (1+ MB each)
- `test_multi_user_result_*.jpg` (2-3 MB each)

**Impact:**
- Adds ~50+ MB of non-source assets
- Images are test artifacts, not documentation assets
- `eval_screenshots/` and `lesson/` are in `.gitignore` but already committed

**Recommendation:**
- Move test images to external storage
- Use `.gitignore` to prevent future commits
- Add only small representative test images if needed for CI

---

### 6. Weak `.gitignore` Configuration

**Current `.gitignore` (inadequate):**
```
venv_311/
dist/
lesson_images/
lesson/
eval_screenshots/
```

**Missing entries:**
- `*.pt`, `*.pth` (model files)
- `*.db`, `*.sqlite` (databases)
- `__pycache__/` (Python cache - directories exist in repo)
- `*.pyc`, `*.pyo`
- `build/` (PyInstaller build)
- `*.spec` (PyInstaller spec file committed: `TuroArnis.spec`)
- `runs/` (YOLO training outputs)
- `*.log` (log files: `app_startup.log` committed)
- Test result images

**Impact:**
- Continued risk of committing generated files
- `__pycache__/` directories present in `app/__pycache__/`, `app/computer_vision/__pycache__/`, etc.

**Recommendation:**
- Expand `.gitignore` with comprehensive Python project template
- Clean existing committed artifacts with `git rm --cached`

---

## Code Quality Issues

### 7. Extensive Debug Print Statements (Not Logging)

**Issue:** Code uses `print()` for debugging instead of proper logging framework.

**Evidence:** 404+ print statements found across codebase.

**Files with heavy print usage:**
- `app/computer_vision/pose_analyzer.py` - Debug prints for every detection step
- `app/computer_vision/gcn_inference.py` - Model loading prints
- `app/app.py` - GUI initialization prints

**Examples:**
```python
print(f"[DEBUG-STICK] Frame shape: {frame.shape}")
print(f"[DEBUG-YOLO] YOLO results count: {len(results_yolo)}")
print(f"[GCN] Loading {viewpoint} model from {resolved_model_path}...")
```

**Impact:**
- Cannot control log levels (DEBUG/INFO/WARNING/ERROR)
- No log rotation or file output
- Production code outputs debug info to console
- Cannot disable debug output without code changes

**Recommendation:**
- Replace with Python `logging` module
- Use appropriate log levels
- Configure handlers for file and console output

---

### 8. Extensive Use of Bare `except:` Clauses

**Issue:** Many exception handlers catch all exceptions without specificity.

**Count:** 228 try/except blocks, many with bare `except:` or `except Exception:`

**Examples:**
- `app/app.py:1047` - `except:` (bare except)
- `app/app.py:1938` - `except:` (bare except)
- `app/main_video.py:436` - `except:` (bare except)
- `app/computer_vision/pose_analyzer.py:800` - `except Exception: return None`
- `app/computer_vision/pose_analyzer.py:823` - `except Exception:`

**Impact:**
- Silently catches KeyboardInterrupt, SystemExit
- Makes debugging difficult (swallows unexpected errors)
- PEP 8 violation (E722)

**Recommendation:**
- Use specific exception types: `except ValueError:`, `except cv2.error:`
- Log exceptions with `logging.exception()` for debugging
- Never use bare `except:` (use `except Exception:` at minimum)

---

### 9. Duplicate Code Across Multiple Files

**Issue:** Significant code duplication between `app.py`, `main_app.py`, and `app_1.py`.

**Evidence:**
- `app/app.py`, `app/main_app.py`, `app/app_1.py` - All contain similar GUI implementations
- `app/main_video.py`, `app/main_image.py` - Similar frame processing logic
- `app/test_image_app.py` - Duplicates visualization functions from `app.py`

**Duplicated patterns:**
- Padding/resizing logic: `cv2.copyMakeBorder(scaled_img, ...)` appears in `main_app.py:457`, `main_video.py:354`, `main_image.py:244`
- Frame overlay copying: `frame.copy()` pattern repeated 20+ times
- Hardcoded color tuples for keypoint rendering

**Impact:**
- Maintenance burden (fix bugs in multiple places)
- Risk of divergent implementations
- Violates DRY principle

**Recommendation:**
- Extract common utilities to shared modules
- Remove obsolete versions (`app_1.py` appears to be backup)
- Create shared visualization module

---

### 10. Hardcoded Configuration Values

**Issue:** Numerous magic numbers and hardcoded thresholds throughout code.

**Examples:**
- `app/computer_vision/pose_analyzer.py:168` - `conf=0.15` (stick detection threshold)
- `app/computer_vision/pose_analyzer.py:319` - `conf=0.3`, `imgsz=480`
- `app/computer_vision/pose_analyzer.py:331` - `conf=0.4`, `imgsz=480`
- `app/computer_vision/pose_analyzer.py:469` - `visibility < 0.3`
- `app/computer_vision/pose_analyzer.py:689` - `stick_len_m = 0.71` (71cm stick length)
- `app/computer_vision/pose_analyzer.py:525` - `frame.shape[1] * 0.25`
- `app/app.py:48` - `SCREEN_WIDTH = 1280`, `SCREEN_HEIGHT = 720`
- `app/computer_vision/gcn_inference.py:199` - `confidence_threshold: 0.50` (default)

**Impact:**
- Difficult to tune parameters
- No centralized configuration
- Values scattered across codebase

**Recommendation:**
- Create centralized config files (JSON/YAML)
- Load configuration at startup
- Document each parameter's purpose

---

### 11. sys.path Manipulation Throughout Codebase

**Issue:** Extensive manual `sys.path` manipulation for imports.

**Count:** 87+ instances of `sys.path.insert` or `sys.path.append`

**Examples:**
```python
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
```

**Files affected:**
- `app/computer_vision/gcn_inference.py:15`
- `app/computer_vision/pose_analyzer.py:12`
- `app/app.py:20-30`
- `app/main_app.py:22-26`
- Most files in `scripts/` directory
- Test files

**Impact:**
- Brittle import structure
- Makes testing difficult
- Violates Python packaging best practices
- PyInstaller-specific hacks scattered throughout

**Recommendation:**
- Use proper package structure with `__init__.py`
- Install package in development mode: `pip install -e .`
- Use relative imports within package
- Single entry point path setup

---

## Architecture Concerns

### 12. Known Classification Failures (Documented in Analysis)

**Issue:** `classification_failure_analysis.md` documents 7 root causes of pose misclassification.

**Critical issues identified:**

**Root Cause #1:** Hypothesis loop selects wrong winner - doesn't verify marginal probability
- **File:** `gcn_inference.py:147-172`
- **Impact:** High - Model predicts wrong technique

**Root Cause #2:** Stick detection failure silently falls back to center
- **File:** `pose_analyzer.py:720-722`
- **Impact:** High - Corrupts all stick features

**Root Cause #3:** Coordinate space mismatch (crop vs frame)
- **File:** `pose_analyzer.py:716-719`, `feature_extraction.py:109-163`
- **Impact:** High - Wrong stick position calculations

**Root Cause #4:** Template standard deviations too wide
- **File:** `app/models/gcn/feature_templates.json`
- **Impact:** Medium - Features don't discriminate

**Root Cause #5:** 2D-only angle calculation (ignores Z depth)
- **File:** `feature_extraction.py:12-19`
- **Impact:** Medium - Cannot distinguish forward vs sideways motion

**Root Cause #6:** MediaPipe in video mode for snapshots
- **File:** `pose_analyzer.py:75`
- **Impact:** Medium - Temporal bleed between frames

**Root Cause #7:** Uniform 0.70 confidence threshold too high
- **File:** `gcn_model_config.json:8,14,20`
- **Impact:** Low - Many valid poses rejected

**Recommendation:**
- Follow the fix priority outlined in the analysis document
- Implement per-class thresholds
- Fix coordinate normalization

---

### 13. GCN Model Loading Without Error Recovery

**Issue:** If GCN models fail to load, the system has no fallback.

**File:** `app/computer_vision/pose_analyzer.py:102-107`

```python
except Exception as e:
    print(f"[ERROR] Could not load GCN models: {e}")
    self.is_gcn = False
    self.gcn_engine = None
    # No fallback to legacy models
    print("[CRITICAL] GCN models failed to load. Pose classification will be unavailable.")
```

**Impact:**
- Application becomes non-functional if models fail to load
- No graceful degradation
- No user-facing error handling

**Recommendation:**
- Add fallback to rule-based or simpler ML approach
- Show user-friendly error dialog
- Allow running in "demo mode" without pose detection

---

## Dependency Issues

### 14. Version Conflicts in Requirements Files

**Issue:** Three different requirements.txt files with conflicting versions.

**Root `requirements.txt`:**
- `torch==2.1.0+cpu`
- `torchvision==0.16.0+cpu`
- `ultralytics==8.3.25`
- `pyinstaller==6.12.0` (listed twice!)

**`deployment_package/requirements.txt`:**
- `torch==2.10.0+cpu` (different version!)
- `torchvision==0.25.0` (different!)
- `ultralytics==8.3.252` (different patch!)
- `opencv-python==4.13.0.92` (root has `>=4.8.0`)

**Impact:**
- PyTorch 2.1.0 vs 2.10.0 mismatch will cause compatibility issues
- Deployment package may behave differently than development
- No single source of truth for dependencies

**Recommendation:**
- Consolidate to single `requirements.txt`
- Use `requirements-dev.txt` and `requirements-prod.txt` if needed
- Pin exact versions for reproducibility
- Use Docker for deployment to ensure consistency

---

### 15. PyInstaller Spec File Committed

**Issue:** `TuroArnis.spec` is committed to repository.

**File:** `TuroArnis.spec` (4 KB)

**Impact:**
- Spec file may contain environment-specific paths
- Should be regenerated via `pyinstaller --onefile` command
- Risk of committing local modifications

**Recommendation:**
- Remove from git: `git rm TuroArnis.spec`
- Add to `.gitignore`
- Document build command in README

---

## Testing Issues

### 16. No Automated Test Suite

**Issue:** Test files present but no test runner configuration.

**Test files found:**
- `test_gcn_integration.py`
- `test_gcn_model_loading.py`
- `app/test_image_app.py`
- `scripts/test_*.py` (multiple)

**Missing:**
- No `pytest.ini` or `setup.cfg`
- No CI/CD pipeline
- No test discovery configuration
- Tests appear to be manual/integration tests, not unit tests

**Impact:**
- No automated regression detection
- Relies on manual testing
- Changes to pose detection may break functionality unnoticed

**Recommendation:**
- Add pytest configuration
- Create unit tests for core logic (feature extraction, GCN inference)
- Set up GitHub Actions for CI
- Separate test data from source code

---

### 17. Test Data Mixed with Source Code

**Issue:** Test images, results, and evaluation data mixed in repository.

**Evidence:**
- `trio/` directory contains 13 test images
- `results_trio/` contains 12 test output images
- `eval_screenshots/` contains evaluation outputs
- Multiple `test_*.jpg` files in root directory

**Impact:**
- Repository bloat
- Unclear which images are test fixtures vs documentation
- Images may be under different licenses

**Recommendation:**
- Move test fixtures to `tests/fixtures/`
- Use Git LFS for binary test data if needed
- Document image sources and licenses

---

## Security Concerns

### 18. Potential Data Exposure in Committed Database

**Issue:** SQLite database committed to repository may contain data.

**Files:**
- `turoarnis.db` (88 KB)
- `app/turoarnis.db` (64 KB)

**Risk:**
- May contain test user data
- May contain personally identifiable information from testing
- No encryption at rest

**Recommendation:**
- Remove databases from git
- Add `.db` to `.gitignore`
- Provide database schema and initialization scripts
- Document data handling procedures

---

### 19. Resource Path Helper Complexity

**Issue:** `get_resource_path()` function handles PyInstaller paths but may have edge cases.

**File:** `app/utils/resource_path.py`

**Risk:**
- Path resolution depends on frozen vs development detection
- May fail if bundled differently
- Windows-specific path handling

**Recommendation:**
- Add comprehensive tests for path resolution
- Document path structure requirements
- Consider using `pkg_resources` or `importlib.resources`

---

## Performance Concerns

### 20. No Performance Benchmarks or Profiling

**Issue:** No evidence of performance testing or optimization tracking.

**Evidence:**
- No benchmark scripts
- No FPS tracking in production code
- No memory usage monitoring
- Image processing pipeline has no timing metrics

**Impact:**
- Performance regressions go unnoticed
- No data to guide optimization efforts
- Unknown if system meets real-time requirements

**Recommendation:**
- Add timing decorators to critical functions
- Log FPS and inference latency
- Create benchmark suite for pose detection pipeline
- Profile memory usage during video processing

---

## Documentation Issues

### 21. Incomplete Project Documentation

**Issue:** README and documentation gaps.

**Evidence:**
- No root-level README.md visible
- Multiple documentation files in `docs/` but organization unclear
- `deployment_package/MANIFEST.md` exists but may be outdated
- `DEVELOPMENT_FLOWCHART.md` present but may not reflect current state

**Recommendation:**
- Create comprehensive README.md at root
- Document setup, build, and deployment processes
- Consolidate scattered documentation
- Remove outdated planning documents

---

### 22. Scattered Planning Documents

**Issue:** Multiple planning/analysis documents in root directory.

**Files:**
- `classification_failure_analysis.md` (12 KB)
- `DEVELOPMENT_FLOWCHART.md`
- `GCN_INTEGRATION_SUMMARY.md`
- `IMPLEMENTATION_PLAN.md`
- `POSE_SELECTION_REMOVAL_PLAN.md`
- `USER_FLOW_COMPARISON.md`

**Impact:**
- Root directory clutter
- Unclear which documents are current vs historical
- `classification_failure_analysis.md` appears to be current (April 2026)

**Recommendation:**
- Move to `docs/planning/` or `.planning/` directory
- Archive completed/obsolete plans
- Keep only current relevant documentation in root

---

## Prioritized Fix Recommendations

### Immediate (Fix This Week)

1. **Expand `.gitignore`** - Add `*.pt`, `*.pth`, `*.db`, `__pycache__/`, `*.spec`, `*.log`, `build/`
2. **Remove committed artifacts** - Run `git rm --cached` for build files, models, DB, venv
3. **Fix requirements.txt conflicts** - Consolidate to single file with consistent versions

### Short Term (Fix This Month)

4. **Implement logging framework** - Replace print statements with proper logging
5. **Fix bare except clauses** - Use specific exception types throughout
6. **Address GCN classification issues** - Follow `classification_failure_analysis.md` fix priority
7. **Extract duplicate code** - Create shared utilities module

### Medium Term (Next Quarter)

8. **Add automated testing** - Set up pytest with CI/CD
9. **Centralize configuration** - Move hardcoded values to config files
10. **Package structure cleanup** - Fix sys.path issues with proper package layout
11. **Add performance monitoring** - Benchmarks and profiling

### Long Term (Ongoing)

12. **Documentation overhaul** - Consolidate and update all docs
13. **Security audit** - Review data handling and path resolution
14. **Model management** - Move to external storage with download scripts

---

*Concerns analysis completed: 2026-04-13*
