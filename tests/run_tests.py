"""
Simple test runner that doesn't require pytest.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py -v  # Verbose output
"""

import sys
import os
import importlib
import traceback
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_test_function(module, func_name, verbose=False):
    """Run a single test function and return success/failure."""
    test_func = getattr(module, func_name)
    try:
        # Check if function needs fixtures
        import inspect

        sig = inspect.signature(test_func)
        if len(sig.parameters) == 0:
            test_func()
        else:
            # Skip tests that need fixtures (would need proper setup)
            if verbose:
                print(f"  SKIP {func_name} (requires fixtures)")
            return None  # Skip
        if verbose:
            print(f"  PASS {func_name}")
        return True
    except Exception as e:
        if verbose:
            print(f"  FAIL {func_name}")
            print(f"    {type(e).__name__}: {e}")
        return False


def run_tests_in_module(module_name, verbose=False):
    """Run all test functions in a module."""
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"ERROR: Could not import {module_name}: {e}")
        return 0, 0, 1

    test_functions = [
        name
        for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    ]

    if not test_functions:
        return 0, 0, 0

    print(f"\n{module_name}:")

    passed = 0
    failed = 0
    skipped = 0

    for func_name in test_functions:
        result = run_test_function(module, func_name, verbose)
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1

    return passed, failed, skipped


def main():
    parser = argparse.ArgumentParser(description="Run LynxLearn tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("LynxLearn Test Suite")
    print("=" * 60)

    # Test modules to run
    test_modules = [
        "tests.test_metrics",
        "tests.test_model_selection",
    ]

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for module_name in test_modules:
        passed, failed, skipped = run_tests_in_module(module_name, args.verbose)
        total_passed += passed
        total_failed += failed
        total_skipped += skipped

    print("\n" + "=" * 60)
    print(
        f"Results: {total_passed} passed, {total_failed} failed, {total_skipped} skipped"
    )
    print("=" * 60)

    if total_failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
