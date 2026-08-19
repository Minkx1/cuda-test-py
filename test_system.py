from __future__ import annotations

import ctypes
import platform
import subprocess


def test_nvidia_smi() -> None:

    print("=" * 70)
    print("NVIDIA-SMI")
    print("=" * 70)

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
            ],
            capture_output=True,
            text=True,
        )

        print(
            result.stdout
        )

        if result.returncode != 0:
            print(
                result.stderr
            )

    except FileNotFoundError:

        print(
            "nvidia-smi was not found."
        )


def test_cuda_driver() -> None:

    print("=" * 70)
    print("CUDA DRIVER")
    print("=" * 70)

    if platform.system() != "Linux":

        print(
            "This low-level test currently "
            "targets Linux."
        )

        return

    try:

        lib = ctypes.CDLL(
            "libcuda.so.1"
        )

        print(
            "libcuda.so.1: OK"
        )

        cu_init = lib.cuInit

        cu_init.argtypes = [
            ctypes.c_uint
        ]

        cu_init.restype = ctypes.c_int

        result = cu_init(0)

        print(
            f"cuInit(): {result}"
        )

        if result != 0:

            raise RuntimeError(
                f"CUDA initialization failed: "
                f"{result}"
            )

        count = ctypes.c_int()

        cu_device_get_count = (
            lib.cuDeviceGetCount
        )

        cu_device_get_count.argtypes = [
            ctypes.POINTER(
                ctypes.c_int
            )
        ]

        cu_device_get_count.restype = (
            ctypes.c_int
        )

        result = (
            cu_device_get_count(
                ctypes.byref(count)
            )
        )

        print(
            f"CUDA device count: "
            f"{count.value}"
        )

        if result != 0:

            raise RuntimeError(
                f"cuDeviceGetCount failed: "
                f"{result}"
            )

    except OSError as exc:

        print(
            "FAILED:"
        )

        print(exc)


if __name__ == "__main__":

    test_nvidia_smi()
    test_cuda_driver()