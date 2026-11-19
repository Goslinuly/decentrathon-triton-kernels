"""bf16 -> fp16 on the GPU gives the same bits as the CPU conversion transformers does."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_gpu_cast_is_bitwise_the_cpu_cast():
    torch.manual_seed(0)
    w = (torch.randn(4096, 1536) * torch.logspace(-8, 4, 1536)).bfloat16()
    w[0, :4] = torch.tensor([float("inf"), -float("inf"), 0.0, -0.0])
    cpu = w.to(device="cuda", dtype=torch.float16)          # what transformers does
    gpu = w.to(device="cuda").to(dtype=torch.float16)       # what we do
    assert torch.equal(cpu.view(torch.int16), gpu.view(torch.int16))
