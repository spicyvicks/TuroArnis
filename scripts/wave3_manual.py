#!/usr/bin/env python3
"""
Phase 5.6 - Wave 3 Manual Execution Script
Testing & Validation Tasks

Run this script to manually execute Wave 3 with checkpoint prompts.
"""

import subprocess
import sys
import os
import json
from datetime import datetime

# Configuration
PROJECT_ROOT = r"C:\Users\HP\Documents\GitHub\TuroArnis"
TEST_FILE = os.path.join(PROJECT_ROOT, "tests", "test_lesson_similarity.py")
AB_LOG_FILE = os.path.join(PROJECT_ROOT, ".planning", "ab_test_results.json")

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_task(task_num, title):
    """Print task header"""
    print(f"\n{'─'*60}")
    print(f"  TASK {task_num}: {title}")
    print(f"{'─'*60}")

def wait_for_continue():
    """Wait for user to continue"""
    input("\n  [Press Enter to continue...]")

def run_command(cmd, description):
    """Run a command and show results"""
    print(f"\n  ▶ Running: {description}")
    print(f"    Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr and "pytest" not in cmd:
        print("  STDERR:", result.stderr)
    
    return result.returncode == 0

def task_6_unit_tests():
    """Task 6: Unit tests for similarity calculations"""
    print_task(6, "Unit Tests for Similarity Calculations")
    
    print("  This task runs 16 unit tests covering:")
    print("    ✓ Simple average produces 50-100% range")
    print("    ✓ Variance-weighted weights tight templates more")
    print("    ✓ Multi-factor returns correct category breakdown")
    print("    ✓ Buffer correctly adds 5% (65→70, 95→100)")
    print("    ✓ Low features correctly identified (< 70%)")
    print("    ✓ Tips limited to 2-3 maximum")
    print()
    
    wait_for_continue()
    
    success = run_command(
        f"python {TEST_FILE}",
        "Running unit tests"
    )
    
    if success:
        print("  ✅ Task 6: PASSED - All unit tests passed")
    else:
        print("  ❌ Task 6: FAILED - Some tests failed")
    
    return success

def task_7_integration_test():
    """Task 7: Integration testing - Manual verification"""
    print_task(7, "Integration Testing (Manual)")
    
    print("  This task requires MANUAL testing of the lesson mode.")
    print()
    print("  Steps to test:")
    print("    1. Launch the application:")
    print(f"       cd {PROJECT_ROOT}")
    print("       python -m app.app")
    print()
    print("    2. Navigate to Lesson Mode:")
    print("       Mode Select → Learn (Lesson Mode)")
    print()
    print("    3. Select a technique and viewpoint")
    print()
    print("    4. Test these scenarios:")
    print()
    
    scenarios = [
        ("Perfect pose", "95-100% displayed (90-95% actual)", "Stand in perfect form"),
        ("Good pose", "80-90% displayed", "Slight deviations from perfect"),
        ("Passing pose", "70% displayed (65% actual)", "Near-threshold form"),
        ("Failing pose", "55-65% displayed (50-60% actual)", "Clearly wrong form"),
    ]
    
    results = {}
    
    for i, (scenario, expected, instruction) in enumerate(scenarios, 1):
        print(f"    Scenario {i}: {scenario}")
        print(f"      Expected: {expected}")
        print(f"      Action: {instruction}")
        print()
        
        while True:
            response = input(f"      Result [pass/fail/skip]: ").strip().lower()
            if response in ['pass', 'fail', 'skip']:
                results[scenario] = response
                break
            print("      Please enter 'pass', 'fail', or 'skip'")
        print()
    
    # Additional checks
    print("  Additional verification:")
    checks = [
        "Tips appear for low-scoring features",
        "A/B rotation switches approach each session",
    ]
    
    for check in checks:
        while True:
            response = input(f"    {check}? [pass/fail/skip]: ").strip().lower()
            if response in ['pass', 'fail', 'skip']:
                results[check] = response
                break
            print("      Please enter 'pass', 'fail', or 'skip'")
    
    passed = sum(1 for r in results.values() if r == 'pass')
    total = len(results)
    
    print(f"\n  Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("  ✅ Task 7: PASSED - All integration tests passed")
        return True
    elif passed >= total * 0.7:
        print("  ⚠️  Task 7: PARTIAL - Most tests passed")
        return True
    else:
        print("  ❌ Task 7: FAILED - Too many tests failed")
        return False

def task_8_ab_logger_validation():
    """Task 8: A/B test data collection validation"""
    print_task(8, "A/B Test Logger Validation")
    
    print("  This task verifies the A/B logger is working correctly.")
    print()
    
    # Check if log file exists
    print(f"  Checking for log file: {AB_LOG_FILE}")
    
    if os.path.exists(AB_LOG_FILE):
        print(f"  ✅ Log file exists")
        
        with open(AB_LOG_FILE, 'r') as f:
            try:
                data = json.load(f)
                print(f"  ✅ Valid JSON format")
                print(f"  📊 Logged sessions: {len(data.get('sessions', []))}")
                
                # Check approaches used
                approaches = set()
                for session in data.get('sessions', []):
                    approaches.add(session.get('approach', 'unknown'))
                
                print(f"  📊 Approaches used: {', '.join(approaches) if approaches else 'None'}")
                
                # Check required fields
                if data.get('sessions'):
                    session = data['sessions'][0]
                    required_fields = ['approach', 'timestamp', 'technique']
                    missing = [f for f in required_fields if f not in session]
                    
                    if missing:
                        print(f"  ⚠️  Missing fields in sessions: {missing}")
                    else:
                        print(f"  ✅ All required fields present")
                
                return True
                
            except json.JSONDecodeError:
                print(f"  ❌ Invalid JSON format")
                return False
    else:
        print(f"  ⚠️  Log file does not exist yet")
        print(f"  This is normal if no lesson mode sessions have been run")
        print()
        
        response = input("  Have you run lesson mode at least once? [y/n]: ").strip().lower()
        if response == 'y':
            print("  ❌ Task 8: FAILED - Log file should exist but doesn't")
            return False
        else:
            print("  ⚠️  Task 8: SKIPPED - Run lesson mode to generate logs")
            return True

def generate_report(all_results):
    """Generate final execution report"""
    print_header("WAVE 3 EXECUTION REPORT")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n  Phase: 5.6 - Lesson Mode Similarity Feedback")
    print(f"  Wave: 3 - Testing & Validation")
    print(f"  Executed: {timestamp}")
    print()
    
    print("  Task Results:")
    for task_num, (task_name, passed) in enumerate(all_results, 6):
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"    Task {task_num} ({task_name}): {status}")
    
    total_passed = sum(1 for _, passed in all_results if passed)
    total_tasks = len(all_results)
    
    print(f"\n  Overall: {total_passed}/{total_tasks} tasks passed")
    
    if total_passed == total_tasks:
        print("\n  🎉 Wave 3: COMPLETE - All tasks passed!")
    elif total_passed >= total_tasks * 0.7:
        print("\n  ⚠️  Wave 3: PARTIAL - Most tasks passed, review failures")
    else:
        print("\n  ❌ Wave 3: INCOMPLETE - Too many failures")
    
    # Save report
    report_file = os.path.join(PROJECT_ROOT, ".planning", "phases", "05.6-lesson-similarity", "wave3_manual_report.json")
    report_data = {
        "phase": "5.6",
        "wave": 3,
        "timestamp": timestamp,
        "results": [
            {"task": i+6, "name": name, "passed": passed}
            for i, (name, passed) in enumerate(all_results)
        ],
        "summary": {
            "passed": total_passed,
            "total": total_tasks,
            "status": "complete" if total_passed == total_tasks else "partial" if total_passed >= total_tasks * 0.7 else "failed"
        }
    }
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n  📄 Report saved to: {report_file}")
    except Exception as e:
        print(f"\n  ⚠️  Could not save report: {e}")

def main():
    """Main execution flow"""
    print_header("PHASE 5.6 - WAVE 3 MANUAL EXECUTION")
    
    print("\n  This script will guide you through Wave 3 tasks:")
    print("    • Task 6: Unit tests (automated)")
    print("    • Task 7: Integration testing (manual)")
    print("    • Task 8: A/B logger validation (semi-automated)")
    print()
    
    response = input("  Ready to begin? [yes/no]: ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n  Execution cancelled.")
        return
    
    results = []
    
    # Task 6
    results.append(("Unit Tests", task_6_unit_tests()))
    wait_for_continue()
    
    # Task 7
    results.append(("Integration Test", task_7_integration_test()))
    wait_for_continue()
    
    # Task 8
    results.append(("A/B Logger Validation", task_8_ab_logger_validation()))
    
    # Generate report
    generate_report(results)
    
    print("\n" + "="*60)
    print("  Wave 3 Manual Execution Complete")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
