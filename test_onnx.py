from __future__ import annotations

import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from runtime_manager import RuntimeManager


MODEL_PATH = "models/test_model.onnx"


def create_model() -> None:

    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 4],
    )

    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 4],
    )

    node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [node],
        "cuda_test",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph,
        producer_name="cuda-runtime-test",
        opset_imports=[
            helper.make_opsetid("", 17)
        ],
    )

    onnx.save(
        model,
        MODEL_PATH,
    )


def main() -> None:

    print()
    print("# ONNX Runtime CUDA test")
    print()

    create_model()

    runtime = RuntimeManager()

    runtime.load("cuda")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print(
        "Session providers:"
    )

    print(
        session.get_providers()
    )

    data = np.array(
        [
            [-1.0, 2.0, -3.0, 4.0]
        ],
        dtype=np.float32,
    )

    start = time.perf_counter()

    result = session.run(
        None,
        {
            "input": data
        },
    )

    elapsed = (
        time.perf_counter() -
        start
    ) * 1000

    print()
    print(
        f"Input:  {data}"
    )

    print(
        f"Output: {result[0]}"
    )

    print(
        f"Time:   {elapsed:.3f} ms"
    )

    print()
    print(
        "[OK] ONNX Runtime inference completed."
    )


if __name__ == "__main__":
    main()