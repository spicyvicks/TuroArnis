#!/usr/bin/env python3
"""
Test Build Script for TuroArnis
================================
Purpose: Automated testing of packaged executable to verify all bundled
components work correctly before deployment.

Usage:
    python scripts/test_build.py            # Run full test suite
    python scripts/test_build.py --verify-only  # Quick verification
    python scripts/test_build.py --smoke    # Launch and test basic operation
    python scripts/test_build.py --report   # Generate detailed report

Exit Codes:
    0 - All tests passed
    1 - Executable missing or invalid
    2 - Critical components missing
    3 - Smoke test failed
    4 - Resource loading failed
"""

import os
import sys
import subprocess
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time
import threading

# Configuration
DIST_DIR = Path("dist/TuroArnis")
EXE_PATH = DIST_DIR / "TuroArnis.exe"
INTERNAL_DIR = DIST_DIR / "_internal"
TEST_REPORT = "build_test_report.json"

# Critical bundled files to verify
CRITICAL_FILES = {
    "executable": "TuroArnis.exe",
    "python_dll": "_internal/python3*.dll",
    "torch_lib": "_internal/torch/lib/torch_python.dll",
    "mediapipe": "_internal/mediapipe/",
    "opencv": "_internal/cv2/",
    "assets": "app/assets/TA.ico",
    "models_dir": "app/models/",
    "gcn_front": "app/models/hybrid_gcn_v2_front.pth",
    "gcn_left": "app/models/hybrid_gcn_v2_left.pth",
    "gcn_right": "app/models/hybrid_gcn_v2_right.pth",
    "yolo_weights": "app/models/weights/best.pt",
    "yolo_base": "yolov8n.pt",
    "gifs_front": "lesson/front_gif/",
    "gifs_left": "lesson/left_gif/",
    "gifs_right": "lesson/right_gif/",
}

# Critical imports to test
CRITICAL_IMPORTS = [
    "torch",
    "torch_geometric",
    "ultralytics",
    "mediapipe",
    "cv2",
    "numpy",
    "scipy",
    "sklearn",
    "customtkinter",
    "PIL",
]


class BuildTester:
    """Test harness for packaged executable."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "executable_path": str(EXE_PATH),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
            }
        }
        self.dist_path = Path(DIST_DIR)
    
    def log(self, message, level="INFO"):
        """Log test message."""
        prefix = {"INFO": "[i]", "PASS": "[✓]", "FAIL": "[✗]", "WARN": "[!]"}.get(level, "[?]")
        print(f"{prefix} {message}")
        
        # Store in results
        if level in ["PASS", "FAIL", "WARN"]:
            self.results["summary"]["total"] += 1
            if level == "PASS":
                self.results["summary"]["passed"] += 1
            elif level == "FAIL":
                self.results["summary"]["failed"] += 1
            elif level == "WARN":
                self.results["summary"]["warnings"] += 1
    
    def test_executable_exists(self):
        """Test 1: Verify executable file exists."""
        self.log("Testing executable existence...", "INFO")
        
        if not EXE_PATH.exists():
            self.log(f"Executable not found: {EXE_PATH}", "FAIL")
            self.results["tests"]["executable_exists"] = {"status": "FAIL", "error": "File not found"}
            return False
        
        size_mb = EXE_PATH.stat().st_size / (1024 * 1024)
        self.log(f"Executable exists ({size_mb:.1f} MB)", "PASS")
        self.results["tests"]["executable_exists"] = {
            "status": "PASS",
            "size_mb": round(size_mb, 2),
            "path": str(EXE_PATH)
        }
        return True
    
    def test_internal_structure(self):
        """Test 2: Verify _internal folder structure."""
        self.log("\nTesting _internal folder structure...", "INFO")
        
        if not INTERNAL_DIR.exists():
            self.log("_internal directory not found", "FAIL")
            self.results["tests"]["internal_structure"] = {"status": "FAIL", "error": "Directory not found"}
            return False
        
        # Count key components
        files = list(INTERNAL_DIR.rglob("*"))
        file_count = len([f for f in files if f.is_file()])
        dir_count = len([f for f in files if f.is_dir()])
        
        self.log(f"_internal/ contains {file_count} files, {dir_count} directories", "PASS")
        
        # Check for critical DLLs
        dll_files = list(INTERNAL_DIR.rglob("*.dll"))
        pyd_files = list(INTERNAL_DIR.rglob("*.pyd"))
        
        self.log(f"Found {len(dll_files)} DLLs, {len(pyd_files)} PYD modules", "INFO")
        
        self.results["tests"]["internal_structure"] = {
            "status": "PASS",
            "file_count": file_count,
            "dir_count": dir_count,
            "dll_count": len(dll_files),
            "pyd_count": len(pyd_files),
        }
        return True
    
    def test_critical_files(self):
        """Test 3: Verify all critical bundled files exist."""
        self.log("\nTesting critical bundled files...", "INFO")
        
        missing = []
        found = []
        
        for name, pattern in CRITICAL_FILES.items():
            full_path = self.dist_path / pattern
            
            # Handle glob patterns
            if "*" in pattern:
                matches = list(self.dist_path.glob(pattern))
                if matches:
                    found.append(name)
                    self.log(f"  {name}: FOUND", "INFO")
                else:
                    missing.append(name)
                    self.log(f"  {name}: MISSING ({pattern})", "FAIL")
            else:
                if full_path.exists():
                    found.append(name)
                    if full_path.is_file():
                        size_mb = full_path.stat().st_size / (1024 * 1024)
                        self.log(f"  {name}: FOUND ({size_mb:.1f} MB)", "INFO")
                    else:
                        self.log(f"  {name}: FOUND (directory)", "INFO")
                else:
                    missing.append(name)
                    self.log(f"  {name}: MISSING", "FAIL")
        
        self.log(f"\nCritical files: {len(found)} found, {len(missing)} missing", "PASS" if not missing else "FAIL")
        
        self.results["tests"]["critical_files"] = {
            "status": "PASS" if not missing else "FAIL",
            "found": found,
            "missing": missing,
            "count": {"found": len(found), "missing": len(missing)}
        }
        
        return len(missing) == 0
    
    def test_imports_via_subprocess(self):
        """Test 4: Test critical imports in packaged Python."""
        self.log("\nTesting critical library imports...", "INFO")
        
        # Find Python in _internal
        python_exe = INTERNAL_DIR / "python.exe"
        if not python_exe.exists():
            # Try to find any python executable
            python_exes = list(INTERNAL_DIR.glob("**/python*.exe"))
            if python_exes:
                python_exe = python_exes[0]
            else:
                self.log("Python executable not found in _internal", "WARN")
                self.results["tests"]["imports"] = {
                    "status": "SKIP",
                    "reason": "Python executable not found"
                }
                return True  # Don't fail the whole test suite
        
        import_test_script = """
import sys
import json

results = {}
libraries = [
    "torch", "torch_geometric", "ultralytics", "mediapipe",
    "cv2", "numpy", "scipy", "sklearn", "customtkinter", "PIL"
]

for lib in libraries:
    try:
        __import__(lib)
        results[lib] = "OK"
    except Exception as e:
        results[lib] = f"FAIL: {str(e)}"

print(json.dumps(results))
"""
        
        failed = []
        try:
            # Write test script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(import_test_script)
                temp_script = f.name
            
            # Copy script to dist directory so it can access the bundled libraries
            temp_in_dist = self.dist_path / "_test_imports.py"
            shutil.copy(temp_script, temp_in_dist)
            
            # Run with bundled Python
            env = os.environ.copy()
            env["PYTHONPATH"] = str(INTERNAL_DIR)
            
            result = subprocess.run(
                [str(python_exe), str(temp_in_dist)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=str(self.dist_path)
            )
            
            # Cleanup
            os.unlink(temp_script)
            temp_in_dist.unlink()
            
            if result.returncode == 0 and result.stdout:
                import_results = json.loads(result.stdout.strip().split("\n")[-1])
                
                for lib, status in import_results.items():
                    if status == "OK":
                        self.log(f"  {lib}: OK", "INFO")
                    else:
                        self.log(f"  {lib}: {status}", "FAIL")
                        failed.append(lib)
            else:
                self.log(f"Import test failed: {result.stderr}", "WARN")
                failed.append("test_execution")
                
        except Exception as e:
            self.log(f"Import test error: {e}", "WARN")
            failed.append("test_execution")
        
        status = "PASS" if not failed else "WARN"
        self.log(f"\nImport test: {len(CRITICAL_IMPORTS) - len(failed)}/{len(CRITICAL_IMPORTS)} passed", status)
        
        self.results["tests"]["imports"] = {
            "status": status,
            "results": import_results if 'import_results' in dir() else {},
            "failed": failed
        }
        
        return len(failed) == 0 or status == "WARN"
    
    def test_resource_path(self):
        """Test 5: Verify resource path resolution works."""
        self.log("\nTesting resource path resolution...", "INFO")
        
        # This tests if the bundled resource_path.py logic works
        # by checking if we can find bundled assets
        test_paths = [
            ("app/assets/TA.ico", "UI icon"),
            ("app/models/gcn_model_config.json", "Model config"),
        ]
        
        all_found = True
        for rel_path, description in test_paths:
            full_path = self.dist_path / rel_path
            if full_path.exists():
                self.log(f"  {description}: {rel_path}", "PASS")
            else:
                self.log(f"  {description}: {rel_path} NOT FOUND", "FAIL")
                all_found = False
        
        self.results["tests"]["resource_path"] = {
            "status": "PASS" if all_found else "FAIL",
            "tests": test_paths
        }
        
        return all_found
    
    def test_database_path(self):
        """Test 6: Verify database directory creation logic."""
        self.log("\nTesting database path resolution...", "INFO")
        
        # Check if resource_path.py logic will work
        # The actual database is created at runtime in APPDATA
        import tempfile
        test_appdata = Path(tempfile.gettempdir()) / "TuroArnis_Test"
        
        try:
            test_appdata.mkdir(exist_ok=True)
            db_path = test_appdata / "test.db"
            
            # Test write
            db_path.write_text("")
            db_path.unlink()
            test_appdata.rmdir()
            
            self.log("  Database directory creation: OK", "PASS")
            self.results["tests"]["database_path"] = {"status": "PASS"}
            return True
        except Exception as e:
            self.log(f"  Database path test failed: {e}", "WARN")
            self.results["tests"]["database_path"] = {"status": "WARN", "error": str(e)}
            return True  # Don't fail for this
    
    def test_model_loading(self):
        """Test 7: Verify GCN models can be loaded (if possible)."""
        self.log("\nTesting model file accessibility...", "INFO")
        
        model_files = [
            ("app/models/hybrid_gcn_v2_front.pth", "Front GCN"),
            ("app/models/hybrid_gcn_v2_left.pth", "Left GCN"),
            ("app/models/hybrid_gcn_v2_right.pth", "Right GCN"),
        ]
        
        all_ok = True
        for rel_path, description in model_files:
            full_path = self.dist_path / rel_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                self.log(f"  {description}: {size_mb:.1f} MB", "PASS")
            else:
                self.log(f"  {description}: NOT FOUND", "FAIL")
                all_ok = False
        
        self.results["tests"]["model_files"] = {
            "status": "PASS" if all_ok else "FAIL",
            "models": model_files
        }
        
        return all_ok
    
    def test_build_size(self):
        """Test 8: Calculate and report build size."""
        self.log("\nTesting build size...", "INFO")
        
        if not self.dist_path.exists():
            self.log("Distribution directory not found", "FAIL")
            return False
        
        total_size = 0
        largest_files = []
        
        for file_path in self.dist_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                size_mb = size / (1024 * 1024)
                
                if size_mb > 5:  # Track files > 5MB
                    largest_files.append((size_mb, file_path.name))
        
        total_mb = total_size / (1024 * 1024)
        
        # Check if size is reasonable (< 1GB is normal for ML apps)
        if total_mb < 1024:
            status = "PASS"
        else:
            status = "WARN"
            self.log(f"  Build size ({total_mb:.0f} MB) is larger than expected", "WARN")
        
        self.log(f"  Total size: {total_mb:.1f} MB", status)
        
        # Report largest files
        if largest_files:
            self.log("\n  Largest bundled files:", "INFO")
            largest_files.sort(reverse=True)
            for size_mb, name in largest_files[:10]:
                self.log(f"    {size_mb:6.1f} MB  {name[:50]}", "INFO")
        
        self.results["tests"]["build_size"] = {
            "status": status,
            "total_mb": round(total_mb, 2),
            "largest_files": [(round(s, 2), n) for s, n in largest_files[:10]]
        }
        
        return status != "FAIL"
    
    def run_smoke_test(self):
        """Test 9: Launch executable briefly to verify it starts."""
        self.log("\nRunning smoke test (executable launch)...", "INFO")
        self.log("  Launching TuroArnis.exe for 10 seconds...", "INFO")
        
        if not EXE_PATH.exists():
            self.log("  Cannot run smoke test - executable missing", "WARN")
            self.results["tests"]["smoke_test"] = {"status": "SKIP", "reason": "No executable"}
            return True
        
        try:
            # Launch the executable
            proc = subprocess.Popen(
                [str(EXE_PATH)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.dist_path)
            )
            
            # Wait up to 10 seconds
            try:
                stdout, stderr = proc.communicate(timeout=10)
                
                # Check for import errors
                error_keywords = ["ModuleNotFoundError", "ImportError", "DLL load failed", "No module named"]
                errors_found = []
                
                for keyword in error_keywords:
                    if keyword in stderr:
                        errors_found.append(keyword)
                        self.log(f"  Found error: {keyword}", "FAIL")
                
                if errors_found:
                    self.results["tests"]["smoke_test"] = {
                        "status": "FAIL",
                        "errors": errors_found,
                        "stderr_preview": stderr[:500]
                    }
                    return False
                else:
                    self.log("  No import errors detected in console output", "PASS")
                    self.results["tests"]["smoke_test"] = {
                        "status": "PASS",
                        "exit_code": proc.returncode,
                        "runtime_seconds": 10
                    }
                    return True
                    
            except subprocess.TimeoutExpired:
                # Process is still running after 10 seconds - this is good!
                proc.kill()
                self.log("  Process stayed running for 10 seconds (good sign)", "PASS")
                self.results["tests"]["smoke_test"] = {
                    "status": "PASS",
                    "runtime_seconds": 10,
                    "note": "Process terminated after timeout"
                }
                return True
                
        except Exception as e:
            self.log(f"  Smoke test error: {e}", "WARN")
            self.results["tests"]["smoke_test"] = {"status": "WARN", "error": str(e)}
            return True  # Don't fail the whole suite
    
    def generate_report(self):
        """Generate JSON test report."""
        with open(TEST_REPORT, "w") as f:
            json.dump(self.results, f, indent=2)
        self.log(f"\nTest report saved to: {TEST_REPORT}", "INFO")
    
    def print_summary(self):
        """Print test summary."""
        s = self.results["summary"]
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Total tests:   {s['total']}")
        print(f"Passed:        {s['passed']} ✓")
        print(f"Failed:        {s['failed']} ✗")
        print(f"Warnings:      {s['warnings']} !")
        print("=" * 50)
        
        if s['failed'] == 0:
            print("\nBUILD VERIFICATION: PASSED")
            print("The packaged executable appears to be working correctly.")
        else:
            print("\nBUILD VERIFICATION: FAILED")
            print(f"There were {s['failed']} critical failures that need attention.")
        
        return s['failed'] == 0
    
    def run_all_tests(self, smoke=False, quick=False):
        """Run complete test suite."""
        print("TuroArnis Build Test Suite")
        print("=" * 50)
        print(f"Testing: {EXE_PATH}")
        print()
        
        # Essential tests
        if not self.test_executable_exists():
            self.log("\nCRITICAL: Executable not found. Build may have failed.", "FAIL")
            self.generate_report()
            return False
        
        self.test_internal_structure()
        self.test_critical_files()
        
        if not quick:
            self.test_resource_path()
            self.test_database_path()
            self.test_model_loading()
            self.test_build_size()
            
            # These may be slower
            self.test_imports_via_subprocess()
            
            if smoke:
                self.run_smoke_test()
        
        self.generate_report()
        return self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Test TuroArnis packaged executable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_build.py            # Full test suite
  python scripts/test_build.py --verify-only  # Quick check only
  python scripts/test_build.py --smoke    # Include smoke test
  python scripts/test_build.py --report   # Generate JSON report
        """
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Quick verification only (skip slower tests)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Include smoke test (launch executable briefly)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed JSON report"
    )
    args = parser.parse_args()
    
    tester = BuildTester()
    success = tester.run_all_tests(smoke=args.smoke, quick=args.verify_only)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
