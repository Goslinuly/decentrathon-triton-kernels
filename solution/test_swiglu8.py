"""SwiGLU with int8-saved activations against PyTorch: exact forward, close backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(1700, 8960), (3, 77, 1000), (5, 64)])
def test_swiglu8_matches_pytorch(shape):
    from solution.swiglu8 import swiglu8

    torch.manual_seed(0)
    gate = (torch.randn(shape, device="cuda") * 2).half().requires_grad_(True)
    up = (torch.randn(shape, device="cuda") * 2).half().requires_grad_(True)
    dout = torch.randn(shape, device="cuda").half()

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(dout.float())
    ref_dg, ref_du = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu8(gate, up)
    out.backward(dout)
    torch.testing.assert_close(out.float(), ref.half().float(), rtol=1e-3, atol=1e-3)
    for got, want in ((gate.grad.float(), ref_dg), (up.grad.float(), ref_du)):
        rel = (got - want).norm() / want.norm()
        assert rel < 1.5e-2, f"backward off by {rel:.2e} (relative norm)"
