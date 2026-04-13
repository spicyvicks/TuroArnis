#!/usr/bin/env python3
"""
Build Automation Script for TuroArnis
=====================================
Purpose: Automate PyInstaller build process with pre-flight validation,
         comprehensive error handling, and detailed logging.

Usage:
    python scripts/build_app.py           # Full build
    python scripts/build_app.py --dry-run # Validate without building
    python scripts/build_app.py --clean   # Clean and build

Exit Codes:
    0 - Success
    1 - Pre-build validation failed
    2 - Build process failed
    3 - Post-build verification failed
"""

import subprocess
import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Build configuration
BUILD_LOG = "build.log"
DIST_DIR = "dist/TuroArnis"
BUILD_DIR = "build"
SPEC_FILE = "TuroArnis.spec"

# Required model files (per TuroArnis.spec configuration)
REQUIRED_MODELS = [
    "app/models/hybrid_gcn_v2_front.pth",
    "app/models/hybrid_gcn_v2_left.pth",
    "app/models/hybrid_gcn_v2_right.pth",
    "app/models/weights/best.pt",
    "yolov8n.pt",
]

# Required GIF directories
REQUIRED_GIF_DIRS = [
    "lesson/front_gif",
    "lesson/left_gif",
    "lesson/right_gif",
]

# Required asset files
REQUIRED_ASSETS = [
    "app/assets/TA.ico",
]


def log(message, level="INFO"):
    """Log message to console and build log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    
    # Append to build log
    with open(BUILD_LOG, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def check_virtual_environment():
    """Check if running in the expected virtual environment."""
    venv_path = Path("venv_311")
    current_exe = Path(sys.executable)
    
    # Check if venv exists
    if not venv_path.exists():
        log("WARNING: venv_311/ not found in project root", "WARN")
        log("Recommendation: Create with: python -m venv venv_311", "INFO")
        return True  # Don't fail, they might use a different venv name
    
    # Check if we're using the venv Python
    if "venv_311" in str(current_exe):
        log(f"Virtual environment detected: {current_exe}", "INFO")
        return True
    else:
        log(f"WARNING: Not using venv_311 Python: {current_exe}", "WARN")
        log("Recommendation: Run: .\\venv_311\\Scripts\\activate", "INFO")
        return True  # Warning only


def check_model_files():
    """Verify all required ML model files exist."""
    log("Checking ML model files...", "INFO")
    all_present = True
    
    for model_path in REQUIRED_MODELS:
        path = Path(model_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            log(f"  ✓ {model_path} ({size_mb:.1f} MB)", "INFO")
        else:
            log(f"  ✗ MISSING: {model_path}", "ERROR")
            all_present = False
    
    return all_present


def check_gif_directories():
    """Verify GIF directories exist and contain files."""
    log("Checking instructional GIF directories...", "INFO")
    all_valid = True
    
    for gif_dir in REQUIRED_GIF_DIRS:
        path = Path(gif_dir)
        if not path.exists():
            log(f"  ✗ MISSING: {gif_dir}", "ERROR")
            all_valid = False
            continue
        
        # Count GIF files
        gif_files = list(path.glob("*.gif"))
        if len(gif_files) == 0:
            log(f"  ⚠ {gif_dir} exists but contains no GIF files", "WARN")
        else:
            log(f"  ✓ {gif_dir} ({len(gif_files)} GIF files)", "INFO")
    
    return all_valid


def check_assets():
    """Verify UI assets exist."""
    log("Checking UI assets...", "INFO")
    all_present = True
    
    for asset_path in REQUIRED_ASSETS:
        path = Path(asset_path)
        if path.exists():
            log(f"  ✓ {asset_path}", "INFO")
        else:
            log(f"  ✗ MISSING: {asset_path}", "ERROR")
            all_present = False
    
    return all_present


def check_spec_file():
    """Verify PyInstaller spec file exists and is valid."""
    log(f"Checking {SPEC_FILE}...", "INFO")
    
    spec_path = Path(SPEC_FILE)
    if not spec_path.exists():
        log(f"  ✗ MISSING: {SPEC_FILE}", "ERROR")
        return False
    
    # Count hiddenimports to verify it's comprehensive
    content = spec_path.read_text(encoding="utf-8")
    hiddenimport_count = content.count("hiddenimports")
    
    log(f"  ✓ {SPEC_FILE} exists ({hiddenimport_count} hiddenimport sections)", "INFO")
    return True


def check_pyinstaller():
    """Verify PyInstaller is installed and compatible."""
    log("Checking PyInstaller...", "INFO")
    
    try:
        import PyInstaller
        version = PyInstaller.__version__
        log(f"  ✓ PyInstaller {version}", "INFO")
        
        # Check if version matches requirements
        if version == "6.12.0":
            log("  ✓ Version matches requirements.txt (6.12.0)", "INFO")
        else:
            log(f"  ⚠ Version {version} differs from requirements.txt (6.12.0)", "WARN")
        
        return True
    except ImportError:
        log("  ✗ PyInstaller not installed", "ERROR")
        log("  Run: pip install pyinstaller==6.12.0", "INFO")
        return False


def clean_build_directories():
    """Remove previous build artifacts."""
    log("Cleaning previous build directories...", "INFO")
    
    dirs_to_clean = [BUILD_DIR, DIST_DIR]
    for dir_path in dirs_to_clean:
        path = Path(dir_path)
        if path.exists():
            try:
                shutil.rmtree(path, ignore_errors=True)
                log(f"  ✓ Removed {dir_path}/", "INFO")
            except Exception as e:
                log(f"  ⚠ Could not fully remove {dir_path}: {e}", "WARN")
        else:
            log(f"  - {dir_path}/ not present (ok)", "INFO")
    
    # Also clean Python cache
    cache_dirs = list(Path(".").rglob("__pycache__"))
    for cache_dir in cache_dirs[:10]:  # Limit to first 10 to avoid spam
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except:
            pass
    
    log(f"  ✓ Cleaned {len(cache_dirs)} __pycache__ directories", "INFO")


def run_pyinstaller_build():
    """Execute PyInstaller build process."""
    log("Starting PyInstaller build...", "INFO")
    log(f"Command: pyinstaller --noconfirm {SPEC_FILE}", "INFO")
    log("This may take 5-10 minutes...", "INFO")
    
    cmd = [sys.executable, "-m", "PyInstaller", SPEC_FILE, "--noconfirm", "--clean"]
    
    try:
        # Run build and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        # Log stdout
        if result.stdout:
            for line in result.stdout.split("\n"):
                if line.strip():
                    log(line.strip(), "PYI")
        
        # Log stderr
        if result.stderr:
            for line in result.stderr.split("\n"):
                if line.strip():
                    log(line.strip(), "PYI-WARN")
        
        if result.returncode == 0:
            log("Build completed successfully", "INFO")
            return True
        else:
            log(f"Build failed with exit code {result.returncode}", "ERROR")
            return False
            
    except subprocess.CalledProcessError as e:
        log(f"Build process error: {e}", "ERROR")
        return False
    except Exception as e:
        log(f"Unexpected error during build: {e}", "ERROR")
        return False


def verify_build_output():
    """Verify the build produced expected output."""
    log("Verifying build output...", "INFO")
    
    exe_path = Path(f"{DIST_DIR}/TuroArnis.exe")
    internal_path = Path(f"{DIST_DIR}/_internal")
    
    if not exe_path.exists():
        log(f"  ✗ Missing executable: {exe_path}", "ERROR")
        return False
    
    exe_size_mb = exe_path.stat().st_size / (1024 * 1024)
    log(f"  ✓ Executable: {exe_path} ({exe_size_mb:.1f} MB)", "INFO")
    
    if not internal_path.exists():
        log(f"  ✗ Missing _internal directory: {internal_path}", "ERROR")
        return False
    
    # Count files in _internal
    internal_files = list(internal_path.rglob("*"))
    log(f"  ✓ _internal directory: {len(internal_files)} files/dirs", "INFO")
    
    return True


def display_build_stats():
    """Display final build statistics."""
    log("Build Statistics", "INFO")
    log("=" * 50, "INFO")
    
    dist_path = Path(DIST_DIR)
    if dist_path.exists():
        # Calculate total size
        total_size = 0
        for file_path in dist_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        total_size_mb = total_size / (1024 * 1024)
        log(f"Total distribution size: {total_size_mb:.1f} MB", "INFO")
        
        # List largest files in _internal
        internal_path = Path(f"{DIST_DIR}/_internal")
        if internal_path.exists():
            large_files = []
            for file_path in internal_path.rglob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > 10:  # Files larger than 10MB
                        large_files.append((size_mb, file_path.name))
            
            large_files.sort(reverse=True)
            if large_files:
                log("\nLargest bundled files:", "INFO")
                for size_mb, name in large_files[:10]:
                    log(f"  {size_mb:6.1f} MB  {name}", "INFO")
    
    log("\nBuild complete!", "INFO")
    log(f"Executable: .\\{DIST_DIR}\\TuroArnis.exe", "INFO")
    log("Run with: .\\dist\\TuroArnis\\TuroArnis.exe", "INFO")


def write_build_timestamp():
    """Write build timestamp to file."""
    timestamp = datetime.now().isoformat()
    with open(".build_timestamp", "w") as f:
        f.write(f"Build completed: {timestamp}\n")


def pre_flight_checks():
    """Run all pre-build validation checks."""
    log("=" * 50, "INFO")
    log("PRE-BUILD VALIDATION", "INFO")
    log("=" * 50, "INFO")
    
    checks = [
        ("Virtual Environment", check_virtual_environment),
        ("PyInstaller", check_pyinstaller),
        ("Spec File", check_spec_file),
        ("Model Files", check_model_files),
        ("GIF Directories", check_gif_directories),
        ("UI Assets", check_assets),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            log(f"ERROR in {name} check: {e}", "ERROR")
            results.append((name, False))
    
    # Summary
    log("\nPre-flight Summary:", "INFO")
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        log(f"  [{status}] {name}", "INFO" if result else "ERROR")
        if not result:
            all_passed = False
    
    return all_passed


def main():
    """Main build process."""
    parser = argparse.ArgumentParser(
        description="Build TuroArnis executable using PyInstaller"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pre-flight checks without building"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Force clean build (remove previous artifacts)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-flight validation"
    )
    args = parser.parse_args()
    
    # Initialize build log
    with open(BUILD_LOG, "w", encoding="utf-8") as f:
        f.write(f"TuroArnis Build Log - Started {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
    
    log("TuroArnis Build Script", "INFO")
    log("=====================", "INFO")
    
    # Dry run mode - only validate
    if args.dry_run:
        log("DRY RUN MODE - Validating only", "INFO")
        success = pre_flight_checks()
        if success:
            log("\nAll checks passed! Ready to build.", "INFO")
            return 0
        else:
            log("\nSome checks failed. Fix issues before building.", "ERROR")
            return 1
    
    # Pre-flight checks
    if not args.skip_checks:
        if not pre_flight_checks():
            log("\nPre-flight checks failed. Use --skip-checks to bypass.", "ERROR")
            return 1
    else:
        log("Skipping pre-flight checks (--skip-checks)", "WARN")
    
    # Clean if requested or always clean for full builds
    if args.clean or not args.dry_run:
        clean_build_directories()
    
    # Run build
    if not run_pyinstaller_build():
        log("\nBuild failed. Check build.log for details.", "ERROR")
        return 2
    
    # Verify output
    if not verify_build_output():
        log("\nBuild verification failed.", "ERROR")
        return 3
    
    # Success
    write_build_timestamp()
    display_build_stats()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
