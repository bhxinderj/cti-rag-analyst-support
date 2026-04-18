"""
Offline test runner.

Discovers every `tests/test_*.py` module and invokes each top-level `test_*`
callable, reporting per-module pass/fail counts and exiting non-zero on the
first failure. Used by `make test` so the suite runs without pytest installed.
"""

from __future__ import annotations

import importlib
import io
import pathlib
import sys
import traceback
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _discover_modules() -> list[str]:
    tests_dir = pathlib.Path(__file__).parent
    return sorted(p.stem for p in tests_dir.glob("test_*.py"))


def _run_unittest(module) -> tuple[int, list[tuple[str, str]]]:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    count = suite.countTestCases()
    if count == 0:
        return 0, []
    buf = io.StringIO()
    runner = unittest.TextTestRunner(stream=buf, verbosity=0)
    result = runner.run(suite)
    failures = [(str(t), tb) for t, tb in result.failures + result.errors]
    return count, failures


def main() -> int:
    modules = _discover_modules()
    total_tests = 0
    failures: list[tuple[str, str, str]] = []

    for mod_name in modules:
        try:
            module = importlib.import_module(f"tests.{mod_name}")
        except Exception:
            failures.append((mod_name, "<import>", traceback.format_exc()))
            print(f"FAIL  tests.{mod_name}  (import error)")
            continue

        mod_failures = 0
        mod_count = 0

        test_names = [n for n in sorted(dir(module)) if n.startswith("test_") and callable(getattr(module, n))]
        for name in test_names:
            mod_count += 1
            try:
                getattr(module, name)()
            except Exception:
                mod_failures += 1
                failures.append((mod_name, name, traceback.format_exc()))

        ut_count, ut_failures = _run_unittest(module)
        mod_count += ut_count
        mod_failures += len(ut_failures)
        for test_id, tb in ut_failures:
            failures.append((mod_name, test_id, tb))

        total_tests += mod_count

        if mod_count == 0:
            print(f"SKIP  tests.{mod_name}  (no tests discovered)")
            continue

        status = "OK  " if mod_failures == 0 else "FAIL"
        print(f"{status}  tests.{mod_name}  ({mod_count} tests)")

    print()
    print(f"Ran {total_tests} tests across {len(modules)} modules")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for mod_name, test_name, tb in failures:
            print(f"\n--- {mod_name}::{test_name} ---")
            print(tb)
        return 1

    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
