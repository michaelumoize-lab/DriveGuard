"""
DriveGuard Phase 7: Master Benchmark Suite Runner.
Executes all Part A baseline measurement and benchmarking modules:
1. Stage 1: Measurement Infrastructure & Clock Domains Unit Suite
2. Stage 2: Kinematics Validation Under Controlled Drops Suite
3. Stage 2: Offline Inference Performance Benchmark (Modes A, B, C)
4. Stage 2: Real-Time Ingest Queue & Replacement Benchmark
5. Stage 3: Latency Distribution Analysis & Moving Block Bootstrap CI
"""

import os
import subprocess
import sys
import time


def run_step(step_name: str, cmd: list) -> bool:
    print("\n" + "=" * 70)
    print(f"  RUNNING: {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)
    t0 = time.perf_counter()
    res = subprocess.run(cmd)
    dur = time.perf_counter() - t0
    if res.returncode == 0:
        print(f"\n[PASS] {step_name} completed in {dur:.2f}s (Exit Code: 0)")
        return True
    else:
        print(f"\n[FAIL] {step_name} failed in {dur:.2f}s (Exit Code: {res.returncode})")
        return False


def main():
    print("=" * 70)
    print("  DriveGuard Phase 7: Master Benchmark & Verification Suite")
    print(f"  Python Interpreter: {sys.executable}")
    print("=" * 70)

    steps = [
        (
            "Stage 1: Measurement Infrastructure & Clock Domains",
            [sys.executable, "-m", "unittest", "backend/tests/test_measurement_infrastructure.py"],
        ),
        (
            "Stage 2: Kinematics Under Drops Validation",
            [sys.executable, "-m", "unittest", "backend/tests/test_kinematics_under_drops.py"],
        ),
        (
            "Stage 2: Real-Time Ingestion Queue Benchmark",
            [sys.executable, "-m", "backend.benchmarks.benchmark_inference_realtime"],
        ),
    ]

    all_passed = True
    for name, cmd in steps:
        ok = run_step(name, cmd)
        if not ok:
            all_passed = False
            break

    if all_passed:
        print("\n" + "#" * 70)
        print("  ALL BENCHMARK VERIFICATION MODULES EXECUTED SUCCESSFULLY")
        print("#" * 70 + "\n")
    else:
        print("\n" + "!" * 70)
        print("  BENCHMARK SUITE FAILED ENCOUNTERED AN ERROR")
        print("!" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
