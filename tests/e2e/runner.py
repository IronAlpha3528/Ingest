"""Opaque-box E2E test runner for ingest pipeline.

Supports executing:
- All tiers (Tiers 1-4)
- Specific tier (--tier 1, --tier 2, etc.)
- Verbose output and structured test reporting.

Usage:
    python tests/e2e/runner.py
    python tests/e2e/runner.py --tier 1
    python -m tests.e2e.runner --tier 4 -v
"""
from __future__ import annotations

import argparse
from io import StringIO
import os
from pathlib import Path
import sys
import time
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.e2e.test_tier1_features import TestTier1Features
from tests.e2e.test_tier2_boundaries import TestTier2Boundaries
from tests.e2e.test_tier3_combinations import TestTier3Combinations
from tests.e2e.test_tier4_scenarios import TestTier4Scenarios

TIER_MAP = {
    1: ("Tier 1: Feature Coverage (Happy Paths)", TestTier1Features),
    2: ("Tier 2: Boundary & Corner Cases", TestTier2Boundaries),
    3: ("Tier 3: Cross-Feature Combinations", TestTier3Combinations),
    4: ("Tier 4: Real-World Application Scenarios", TestTier4Scenarios),
}


def build_suite(tier: int | None = None) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    if tier:
        if tier not in TIER_MAP:
            raise ValueError(f"Invalid tier {tier}. Valid options: 1, 2, 3, 4")
        _, test_cls = TIER_MAP[tier]
        suite.addTests(loader.loadTestsFromTestCase(test_cls))
    else:
        for _, test_cls in TIER_MAP.values():
            suite.addTests(loader.loadTestsFromTestCase(test_cls))
    return suite


def run_e2e_suite(tier: int | None = None, verbose: bool = False) -> bool:
    print("=" * 70)
    print("  INGEST E2E TEST RUNNER — 4-TIER OPAQUE-BOX VERIFICATION")
    print("=" * 70)

    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()

    total_passed = 0
    total_failed = 0
    total_errors = 0

    tiers_to_run = [tier] if tier else [1, 2, 3, 4]

    for t in tiers_to_run:
        tier_title, test_cls = TIER_MAP[t]
        print(f"\n>>> Running {tier_title}...")
        sub_suite = unittest.TestLoader().loadTestsFromTestCase(test_cls)
        res = runner.run(sub_suite)

        num_tests = sub_suite.countTestCases()
        num_err = len(res.errors)
        num_fail = len(res.failures)
        num_pass = num_tests - num_err - num_fail

        total_passed += num_pass
        total_failed += num_fail
        total_errors += num_err

        status_str = "PASS" if (num_err == 0 and num_fail == 0) else "FAIL"
        print(f"    Result: [{status_str}] — Passed: {num_pass}/{num_tests}, Failed: {num_fail}, Errors: {num_err}")

    elapsed = time.time() - start_time
    total_tests = total_passed + total_failed + total_errors

    print("\n" + "=" * 70)
    print("  E2E TEST EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Total Test Cases : {total_tests}")
    print(f"  Passed           : {total_passed}")
    print(f"  Failed           : {total_failed}")
    print(f"  Errors           : {total_errors}")
    print(f"  Total Duration   : {elapsed:.3f} seconds")
    print("=" * 70)

    if total_failed == 0 and total_errors == 0:
        print("  OVERALL VERDICT: ALL TESTS PASSED (100% SUCCESS)")
        print("=" * 70 + "\n")
        return True
    else:
        print("  OVERALL VERDICT: TEST FAILURES DETECTED")
        print("=" * 70 + "\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Ingest E2E Test Suite (Tiers 1-4)")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run a specific test tier only (1, 2, 3, or 4)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose test reporting")
    args = parser.parse_args()

    success = run_e2e_suite(tier=args.tier, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
