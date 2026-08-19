from __future__ import annotations

import time

import numpy as np

from faster_whisper import WhisperModel

from runtime_manager import RuntimeManager


MODEL = "tiny"


def main() -> None:

    print()
    print("# faster-whisper CUDA test")
    print()

    runtime = RuntimeManager()

    runtime.load("cuda")

    print()
    print(
        "[Whisper] Loading model..."
    )

    start = time.perf_counter()

    model = WhisperModel(
        MODEL,
        device="cuda",
        compute_type="float16",
    )

    elapsed = (
        time.perf_counter() -
        start
    )

    print(
        f"[Whisper] Model loaded "
        f"in {elapsed:.2f}s"
    )

    # 1 second of silence.
    audio = np.zeros(
        16000,
        dtype=np.float32,
    )

    print()
    print(
        "[Whisper] Running inference..."
    )

    start = time.perf_counter()

    segments, info = model.transcribe(
        audio,
        beam_size=1,
    )

    segments = list(segments)

    elapsed = (
        time.perf_counter() -
        start
    )

    print(
        f"[Whisper] Inference: "
        f"{elapsed:.3f}s"
    )

    print(
        f"[Whisper] Language: "
        f"{info.language}"
    )

    print(
        f"[Whisper] Probability: "
        f"{info.language_probability}"
    )

    print()

    for segment in segments:

        print(
            f"[{segment.start:.2f} -> "
            f"{segment.end:.2f}] "
            f"{segment.text}"
        )

    print()
    print(
        "[OK] faster-whisper CUDA test completed."
    )


if __name__ == "__main__":
    main()