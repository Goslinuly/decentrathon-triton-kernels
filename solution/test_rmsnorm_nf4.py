"""RMSNorm and NF4 dequantization against the code they replace. Run on a GPU."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference_rmsnorm(x, weight, eps):
    # transformers' Qwen2RMSNorm.forward
    input_dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(input_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1000)])
def test_rmsnorm_matches_transformers(dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 3).to(dtype).requires_grad_(True)
    weight = torch.rand(shape[-1], device="cuda") + 0.5           # fp32, frozen
    dy = torch.randn(shape, device="cuda")

    ref = reference_rmsnorm(x, weight, 1e-6)
    ref.backward(dy.to(ref.dtype))
    ref_dx, x.grad = x.grad.clone(), None

    out = rmsnorm(x, weight, 1e-6)
    out.backward(dy.to(out.dtype))
    assert out.dtype == ref.dtype
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    # The backward reads an fp32 input back from its fp16 copy: ~5e-4 relative per element.
    rel = (x.grad.float() - ref_dx.float()).norm() / ref_dx.float().norm()
    assert rel < 3e-3, f"backward off by {rel:.2e} (relative norm)"


@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (256, 8960), (100, 64)])
def test_nf4_dequant_matches_bitsandbytes(shape):
    import bitsandbytes.functional as bnbf
    from solution.nf4 import nf4_dequant

    torch.manual_seed(0)
    w = (torch.randn(shape, device="cuda") * 0.02).half()
    packed, state = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4",
                                       compress_statistics=True)
    ref = bnbf.dequantize_4bit(packed, state)
    s2 = state.state2
    ours = nf4_dequant(packed, state.absmax, s2.absmax, s2.code, state.offset, state.blocksize,
                       s2.blocksize, state.shape, state.dtype)
    assert ours.shape == ref.shape and ours.dtype == ref.dtype
    torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-6)


def test_nf4_linear_fast_path_matches_bitsandbytes():
    import bitsandbytes as bnb
    from solution.nf4 import NF4Linear, _prepared

    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(1536, 8960, bias=False, compute_dtype=torch.float16,
                              quant_type="nf4", compress_statistics=True)
    layer.weight = bnb.nn.Params4bit(data=(torch.randn(8960, 1536) * 0.02).half(),
                                     requires_grad=False, compress_statistics=True,
                                     quant_type="nf4")
    layer = layer.cuda()
    x = torch.randn(2000, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(2000, 8960, device="cuda", dtype=torch.float16)

    ref = bnb.matmul_4bit(x, layer.weight, quant_state=layer.weight.quant_state)
    ref.backward(grad)
    ref_dx, x.grad = x.grad.clone(), None

    out = NF4Linear.apply(x, _prepared(layer))
    out.backward(grad)
    # bitsandbytes' own dequantization here vs ours: equal up to fp32 rounding of the scales.
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref_dx, rtol=2e-3, atol=2e-3)
