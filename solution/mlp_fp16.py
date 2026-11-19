"""The MLP's input cast to fp16 once, instead of a round trip through fp32 per projection.

The residual stream is fp32 in this run, so each 4-bit projection of the MLP gets an fp32 input:
bitsandbytes' `Linear4bit` casts it to its fp16 compute dtype, multiplies, and casts the
[tokens, 8960] result back to fp32 (`.to(inp_dtype)`). The SwiGLU then reads and keeps fp32
gate/up tensors for its backward, and `down_proj` casts its input to fp16 again; the backward
does the same in reverse.

The values do not change when the input is cast once up front: the projections multiply the
same fp16 numbers, the fp32 copies of their outputs were exact, and the SwiGLU computes in fp32
inside its kernel and rounds to fp16 exactly where `down_proj` used to. What goes away are the
casts and half of the MLP's saved activations. Plain torch, around the kernels.
"""

import torch


def _fp16_compute(module):
    return getattr(module, "compute_dtype", None) == torch.float16


def patch_mlp_fp16():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    inner = cls.forward

    def forward(self, x):
        if (torch.is_tensor(x) and x.is_cuda and x.dtype == torch.float32
                and all(_fp16_compute(getattr(self, name, None))
                        for name in ("gate_proj", "up_proj", "down_proj"))):
            x = x.to(torch.float16)
        return inner(self, x)

    cls.forward = forward
