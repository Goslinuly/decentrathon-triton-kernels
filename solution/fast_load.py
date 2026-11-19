"""Model loading: the checkpoint's bf16 -> fp16 conversion done on the GPU instead of the CPU.

transformers 5.16.1 materializes every checkpoint tensor with `tensor.to(device="cuda",
dtype=float16)` (`core_model_loading._materialize_copy`). For a CPU -> GPU copy that also changes
the dtype, PyTorch converts on the CPU first, so the ~3 GB of bf16 weights are converted by the
Kaggle VM's few vCPUs, one tensor after another (loading with on-the-fly 4-bit quantization is
sequential). Here the bf16 tensor is copied to the GPU as is and converted there. bf16 -> fp16 is
IEEE round-to-nearest-even on both, so every weight is bit for bit the same; only where the
conversion runs changes. Anything else (other devices, no dtype change) takes the original.
"""

import torch


def patch_fast_load():
    try:
        from transformers import core_model_loading as cml
    except ImportError:
        return
    original = getattr(cml, "_materialize_copy", None)
    if original is None or getattr(original, "_gpu_cast", False):
        return

    def materialize_copy(tensor, device=None, dtype=None):
        tensor = tensor[...]            # reads the slice out of the safetensors file
        if (dtype is not None and device is not None and tensor.device.type == "cpu"
                and torch.device(device).type == "cuda" and tensor.dtype != dtype
                and tensor.is_floating_point() and dtype.is_floating_point):
            return tensor.to(device=device).to(dtype=dtype)
        if dtype is not None or device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    materialize_copy._gpu_cast = True
    cml._materialize_copy = materialize_copy
