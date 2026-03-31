"""Run narrow repository smoke checks without requiring pytest."""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TEST_MODULES = [
    "tests.test_cti_representation",
    "tests.test_eval_query_loading",
    "tests.test_evaluation_trace",
    "tests.test_misp_parser",
    "tests.test_rag_grounding",
    "tests.test_retrieval_quality",
    "tests.test_retrieval_trace",
    "tests.test_setup_and_bm25_artifacts",
    "tests.test_snapshot_logic",
]


def _run_unittest_cases(module) -> tuple[bool, int]:
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    if suite.countTestCases() == 0:
        return True, 0

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful(), result.testsRun


def _run_function_tests(module) -> tuple[bool, list[str]]:
    executed: list[str] = []

    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue

        candidate = getattr(module, name)
        if not callable(candidate):
            continue

        signature = inspect.signature(candidate)
        if signature.parameters:
            continue

        try:
            candidate()
        except Exception:
            traceback.print_exc()
            return False, executed

        executed.append(f"{module.__name__}.{name}")

    return True, executed


def main() -> int:
    total_unittest_cases = 0
    total_function_tests = 0

    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)

        ok, cases_run = _run_unittest_cases(module)
        total_unittest_cases += cases_run
        if not ok:
            return 1

        ok, executed = _run_function_tests(module)
        total_function_tests += len(executed)
        if not ok:
            return 1

    print(
        f"Smoke checks passed: {total_unittest_cases} unittest case(s), "
        f"{total_function_tests} standalone test(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
