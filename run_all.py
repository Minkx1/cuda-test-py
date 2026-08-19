from __future__ import annotations

import subprocess
import sys


TESTS = [
    "test_system.py",
    "test_onnx.py",
    "test_whisper.py",
    "test_llama.py",
]


def main() -> None:

    print()
    print(
        "=========================================="
    )

    print(
        " CUDA Runtime Compatibility Test"
    )

    print(
        "=========================================="
    )

    print()

    for test in TESTS:

        print()
        print(
            "=" * 70
        )

        print(
            f"RUNNING: {test}"
        )

        print(
            "=" * 70
        )

        result = subprocess.run(
            [
                sys.executable,
                test,
            ]
        )

        if result.returncode != 0:

            print()
            print(
                f"[FAIL] {test}"
            )

            print(
                "Stopping test suite."
            )

            sys.exit(
                result.returncode
            )

        print()
        print(
            f"[PASS] {test}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "ALL TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()