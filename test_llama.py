from __future__ import annotations

import os
import time

from llama_cpp import Llama

from runtime_manager import RuntimeManager


MODEL_PATH = "models/model.gguf"


def main() -> None:

    print()
    print("# llama.cpp CUDA test")
    print()

    runtime = RuntimeManager()

    runtime.load("cuda")

    print()
    print(
        "[Llama] Loading model..."
    )

    if not os.path.exists(MODEL_PATH):

        raise RuntimeError(
            f"GGUF model not found:\n"
            f"  {MODEL_PATH}\n\n"
            "Upload a small GGUF model to "
            "models/model.gguf before running "
            "this test."
        )

    start = time.perf_counter()

    llm = Llama(
        model_path=MODEL_PATH,

        # Important:
        # This tells llama.cpp to offload layers.
        n_gpu_layers=-1,

        # Keep the test small.
        n_ctx=512,

        verbose=True,
    )

    elapsed = (
        time.perf_counter() -
        start
    )

    print(
        f"[Llama] Loaded in "
        f"{elapsed:.2f}s"
    )

    print()
    print(
        "[Llama] Running inference..."
    )

    output = llm(
        "Say hello in one short sentence.",
        max_tokens=32,
        temperature=0.0,
    )

    print()
    print(
        output["choices"][0]["text"]
    )

    print()
    print(
        "[OK] llama.cpp CUDA test completed."
    )


if __name__ == "__main__":
    main()