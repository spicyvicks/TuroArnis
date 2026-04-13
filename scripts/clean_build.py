#!/usr/bin/env python3
"""
Clean Build Script for TuroArnis
================================
Purpose: Remove PyInstaller build artifacts and temporary files.

Usage:
    python scripts/clean_build.py           # Clean standard artifacts
    python scripts/clean_build.py --all     # Clean + remove build_timestamp
    python scripts/clean_build.py --models  # Also remove model files (careful!)

What gets removed:
    - build/ directory (PyInstaller work files)
    - dist/ directory (executable output)
    - __pycache__/ directories (Python bytecode)
    - *.pyc, *.pyo files (compiled Python)
    - build.log (build output log)
    - .build_timestamp (build marker)
    - *.spec (spec file - only if --all)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def log(message):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def remove_directory(path, description):
    """Safely remove a directory tree."""
    if path.exists():
        try:
            shutil.rmtree(path, ignore_errors=True)
            log(f"✓ Removed {description}: {path}")
            return True
        except Exception as e:
            log(f"✗ Error removing {description}: {e}")
            return False
    else:
        log(f"- {description} not found: {path}")
        return True


def remove_file(path, description):
    """Safely remove a file."""
    if path.exists():
        try:
            path.unlink()
            log(f"✓ Removed {description}: {path}")
            return True
        except Exception as e:
            log(f"✗ Error removing {description}: {e}")
            return False
    else:
        log(f"- {description} not found: {path}")
        return True


def clean_pycache():
    """Remove all __pycache__ directories."""
    log("\nCleaning Python cache directories...")
    count = 0
    failed = 0
    
    for pycache in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(pycache, ignore_errors=True)
            count += 1
        except:
            failed += 1
    
    if count > 0:
        log(f"✓ Removed {count} __pycache__ directories")
    if failed > 0:
        log(f"⚠ Failed to remove {failed} __pycache__ directories")
    
    return failed == 0


def clean_pyc_files():
    """Remove .pyc and .pyo files."""
    log("\nCleaning compiled Python files...")
    count = 0
    
    for pattern in ["*.pyc", "*.pyo"]:
        for file_path in Path(".").rglob(pattern):
            try:
                file_path.unlink()
                count += 1
            except:
                pass
    
    if count > 0:
        log(f"✓ Removed {count} .pyc/.pyo files")
    else:
        log("- No .pyc/.pyo files found")
    
    return True


def clean_build_artifacts():
    """Remove PyInstaller build artifacts."""
    log("\nCleaning PyInstaller build artifacts...")
    
    success = True
    
    # Main build directories
    success &= remove_directory(Path("build"), "build directory")
    success &= remove_directory(Path("dist"), "dist directory")
    
    # Build log
    success &= remove_file(Path("build.log"), "build log")
    
    # Build timestamp
    success &= remove_file(Path(".build_timestamp"), "build timestamp")
    
    return success


def clean_spec_file():
    """Remove .spec file (use with caution)."""
    log("\nCleaning spec file...")
    return remove_file(Path("TuroArnis.spec"), "spec file")


def clean_model_files():
    """Remove model files (large binaries)."""
    log("\nCleaning model files...")
    log("WARNING: This will remove trained model files!", "WARN")
    
    models = [
        Path("app/models/hybrid_gcn_v2_front.pth"),
        Path("app/models/hybrid_gcn_v2_left.pth"),
        Path("app/models/hybrid_gcn_v2_right.pth"),
        Path("app/models/weights/best.pt"),
        Path("yolov8n.pt"),
    ]
    
    success = True
    for model in models:
        success &= remove_file(model, "model file")
    
    return success


def get_size_summary():
    """Calculate size of build artifacts."""
    total_size = 0
    
    for dir_name in ["build", "dist"]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
    
    # Add cache directories
    for pycache in Path(".").rglob("__pycache__"):
        for file_path in pycache.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    
    mb = total_size / (1024 * 1024)
    return mb


def main():
    parser = argparse.ArgumentParser(
        description="Clean TuroArnis build artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/clean_build.py           # Standard clean
  python scripts/clean_build.py --all     # Thorough clean including spec
  python scripts/clean_build.py --dry-run # Show what would be removed
        """
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove .spec file and build timestamp"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="WARNING: Also remove trained model files (large .pth/.pt)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing"
    )
    args = parser.parse_args()
    
    log("TuroArnis Clean Build Script")
    log("=" * 40)
    
    if args.dry_run:
        log("DRY RUN MODE - Showing what would be removed")
        size_mb = get_size_summary()
        log(f"Estimated space to recover: {size_mb:.1f} MB")
        log("\nWould remove:")
        log("  - build/ directory")
        log("  - dist/ directory")
        log("  - __pycache__/ directories")
        log("  - *.pyc, *.pyo files")
        log("  - build.log")
        log("  - .build_timestamp")
        if args.all:
            log("  - TuroArnis.spec (use with caution)")
        if args.models:
            log("  - Model files (.pth, .pt) - CAREFUL!")
        return 0
    
    # Show current size
    size_before = get_size_summary()
    if size_before > 0:
        log(f"Current artifact size: {size_before:.1f} MB")
    
    # Confirm model deletion
    if args.models:
        log("\n" + "!" * 50)
        log("WARNING: You are about to delete trained ML models!")
        log("These files take time to recreate/retrain.")
        log("!" * 50 + "\n")
        
        if sys.stdin.isatty():  # Interactive mode
            response = input("Type 'DELETE' to confirm model removal: ")
            if response != "DELETE":
                log("Model deletion cancelled.")
                args.models = False
    
    # Perform cleaning
    all_success = True
    
    # Always clean these
    all_success &= clean_build_artifacts()
    all_success &= clean_pycache()
    all_success &= clean_pyc_files()
    
    # Optional cleaning
    if args.all:
        all_success &= clean_spec_file()
    
    if args.models:
        all_success &= clean_model_files()
    
    # Show results
    size_after = get_size_summary()
    saved_mb = size_before - size_after
    
    log("\n" + "=" * 40)
    log(f"Space recovered: {saved_mb:.1f} MB")
    
    if all_success:
        log("Clean completed successfully")
        return 0
    else:
        log("Clean completed with some errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
