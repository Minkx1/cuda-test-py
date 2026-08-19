from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal


Device = Literal["cuda", "cpu", "auto"]


class RuntimeManager:
    """
    Experimental application-local CUDA runtime manager.

    The goal is to test whether CUDA-dependent Python packages can use
    libraries stored inside:

        runtime/cuda/

    instead of relying on a system-wide CUDA Toolkit installation.
    """

    def __init__(
        self,
        runtime_dir: Path | str = "runtime",
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.cuda_dir = self.runtime_dir / "cuda"

        self.device: str | None = None

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def load(self, device: Device) -> None:
        if device not in ("cuda", "cpu", "auto"):
            raise ValueError(
                f"Invalid device: {device!r}"
            )

        print()
        print("=" * 70)
        print(f" RuntimeManager.load({device!r})")
        print("=" * 70)

        # ---------------------------------------------------------------
        # CPU
        # ---------------------------------------------------------------

        if device == "cpu":
            print("[Runtime] CPU explicitly requested.")
            print("[Runtime] CUDA initialization skipped.")

            self.device = "cpu"
            return

        # ---------------------------------------------------------------
        # CUDA / AUTO
        # ---------------------------------------------------------------

        try:
            self._check_cuda_driver()

            print("[Runtime] NVIDIA CUDA driver: OK")

            self._prepare_cuda_environment()

            print("[Runtime] CUDA runtime: OK")

            self._configure_library_path()

            print("[Runtime] Library path configured.")

            self._preload_onnx_runtime_cuda()

            print("[Runtime] CUDA libraries preloaded.")

            self._verify_onnx_cuda()

            print("[Runtime] ONNX Runtime CUDA provider: OK")

            self.device = "cuda"

            print()
            print("[Runtime] CUDA initialization SUCCESS")
            print()

        except Exception as exc:
            if device == "cuda":
                raise RuntimeError(
                    "CUDA was explicitly requested, "
                    "but initialization failed."
                ) from exc

            print()
            print(
                "[Runtime] CUDA initialization failed:"
            )
            print(f"  {type(exc).__name__}: {exc}")
            print()
            print("[Runtime] Falling back to CPU.")

            self.device = "cpu"

    # =====================================================================
    # STEP 1
    # =====================================================================

    def _check_cuda_driver(self) -> None:
        """
        Check NVIDIA CUDA Driver API.

        Important:

        This checks the DRIVER, not the CUDA Toolkit.

        CUDA Toolkit:
            libcudart.so
            libcublas.so
            ...

        NVIDIA driver:
            libcuda.so.1
        """

        system = platform.system()

        if system == "Linux":
            try:
                driver = ctypes.CDLL("libcuda.so.1")
            except OSError as exc:
                raise RuntimeError(
                    "Could not load libcuda.so.1. "
                    "NVIDIA driver is probably unavailable."
                ) from exc

        elif system == "Windows":
            try:
                driver = ctypes.WinDLL("nvcuda.dll")
            except OSError as exc:
                raise RuntimeError(
                    "Could not load nvcuda.dll. "
                    "NVIDIA driver is probably unavailable."
                ) from exc

        else:
            raise RuntimeError(
                f"Unsupported OS: {system}"
            )

        # CUresult cuInit(unsigned int Flags)

        cu_init = driver.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int

        result = cu_init(0)

        if result != 0:
            raise RuntimeError(
                f"cuInit() failed with CUDA error {result}"
            )

        # CUresult cuDeviceGetCount(int *count)

        cu_device_get_count = driver.cuDeviceGetCount

        cu_device_get_count.argtypes = [
            ctypes.POINTER(ctypes.c_int)
        ]

        cu_device_get_count.restype = ctypes.c_int

        count = ctypes.c_int()

        result = cu_device_get_count(
            ctypes.byref(count)
        )

        if result != 0:
            raise RuntimeError(
                "cuDeviceGetCount() failed "
                f"with CUDA error {result}"
            )

        if count.value == 0:
            raise RuntimeError(
                "CUDA driver is present, "
                "but no CUDA GPU was detected."
            )

        print(
            f"[Runtime] CUDA devices detected: {count.value}"
        )

    # =====================================================================
    # STEP 2
    # =====================================================================

    def _prepare_cuda_environment(self) -> None:
        """
        Prepare runtime/cuda/.

        We use NVIDIA's official Python runtime packages.

        We do NOT install them into the normal environment.

        Instead, their native libraries are copied into:

            runtime/cuda/
        """

        self.cuda_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self._has_cuda_libraries():
            print(
                "[Runtime] Existing CUDA libraries found."
            )
            return

        print(
            "[Runtime] CUDA libraries are missing."
        )

        self._download_cuda_packages()

    def _has_cuda_libraries(self) -> bool:
        if platform.system() == "Linux":

            required = [
                "libcudart.so*",
                "libcublas.so*",
                "libcudnn.so*",
            ]

        elif platform.system() == "Windows":

            required = [
                "cudart64_*.dll",
                "cublas64_*.dll",
                "cudnn*.dll",
            ]

        else:
            return False

        for pattern in required:

            matches = list(
                self.cuda_dir.glob(pattern)
            )

            if not matches:
                return False

        return True

    def _download_cuda_packages(self) -> None:

        packages = [
            "nvidia-cuda-runtime-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12",
        ]

        temp_dir = (
            self.runtime_dir /
            "_nvidia_packages"
        )

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "[Runtime] Downloading NVIDIA packages..."
        )

        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(temp_dir),
            *packages,
        ]

        print(
            "$",
            " ".join(command),
        )

        result = subprocess.run(
            command,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Failed to download NVIDIA CUDA packages."
            )

        self._collect_native_libraries(
            temp_dir
        )

        shutil.rmtree(temp_dir)

    def _collect_native_libraries(
        self,
        source: Path,
    ) -> None:

        if platform.system() == "Linux":
            patterns = (
                "*.so",
                "*.so.*",
            )

        elif platform.system() == "Windows":
            patterns = (
                "*.dll",
            )

        else:
            raise RuntimeError(
                "Unsupported OS."
            )

        libraries: list[Path] = []

        for pattern in patterns:
            libraries.extend(
                source.rglob(pattern)
            )

        # Deduplicate
        seen: set[Path] = set()

        for library in libraries:

            resolved = library.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)

            destination = (
                self.cuda_dir /
                library.name
            )

            print(
                f"[Runtime] Copying "
                f"{library.name}"
            )

            shutil.copy2(
                library,
                destination,
            )

        if not libraries:
            raise RuntimeError(
                "No native CUDA libraries "
                "were found in NVIDIA packages."
            )

    # =====================================================================
    # STEP 3
    # =====================================================================

    def _configure_library_path(self) -> None:
        """
        Make runtime/cuda visible to the dynamic loader.

        Linux:
            LD_LIBRARY_PATH

        Windows:
            os.add_dll_directory()
        """

        cuda_path = str(
            self.cuda_dir.resolve()
        )

        if platform.system() == "Linux":

            current = os.environ.get(
                "LD_LIBRARY_PATH",
                "",
            )

            paths = [
                cuda_path,
                current,
            ]

            os.environ[
                "LD_LIBRARY_PATH"
            ] = ":".join(
                p for p in paths if p
            )

            print(
                "[Runtime] LD_LIBRARY_PATH:"
            )

            print(
                os.environ[
                    "LD_LIBRARY_PATH"
                ]
            )

        elif platform.system() == "Windows":

            if hasattr(
                os,
                "add_dll_directory",
            ):
                os.add_dll_directory(
                    cuda_path
                )

    # =====================================================================
    # STEP 4
    # =====================================================================

    def _preload_onnx_runtime_cuda(self) -> None:

        import onnxruntime as ort

        if not hasattr(
            ort,
            "preload_dlls",
        ):
            raise RuntimeError(
                "This ONNX Runtime version "
                "does not provide preload_dlls()."
            )

        print(
            "[Runtime] Calling "
            "onnxruntime.preload_dlls()..."
        )

        ort.preload_dlls(
            directory=str(
                self.cuda_dir.resolve()
            )
        )

    # =====================================================================
    # STEP 5
    # =====================================================================

    def _verify_onnx_cuda(self) -> None:

        import onnxruntime as ort

        providers = (
            ort.get_available_providers()
        )

        print(
            "[Runtime] ONNX Runtime providers:"
        )

        for provider in providers:
            print(
                f"    {provider}"
            )

        if (
            "CUDAExecutionProvider"
            not in providers
        ):
            raise RuntimeError(
                "CUDAExecutionProvider "
                "is not available."
            )

    # =====================================================================
    # INFORMATION
    # =====================================================================

    def status(self) -> None:

        print()
        print("=" * 70)
        print(" Runtime status")
        print("=" * 70)

        print(
            f"OS:          {platform.system()}"
        )

        print(
            f"Architecture: {platform.machine()}"
        )

        print(
            f"Device:      {self.device}"
        )

        print(
            f"CUDA dir:    "
            f"{self.cuda_dir.resolve()}"
        )

        if self.cuda_dir.exists():

            files = list(
                self.cuda_dir.iterdir()
            )

            print(
                f"Libraries:   {len(files)}"
            )

            for file in files:
                print(
                    f"    {file.name}"
                )

        try:

            import onnxruntime as ort

            print(
                "ORT providers:"
            )

            for provider in (
                ort.get_available_providers()
            ):
                print(
                    f"    {provider}"
                )

        except Exception:
            pass

        print("=" * 70)