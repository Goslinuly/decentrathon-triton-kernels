"""One varlen attention call against SDPA run per sequence (the path it replaces)."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_varlen_matches_per_sequence_sdpa():
    from solution.varlen_attention import varlen_attention

    torch.manual_seed(0)
    lengths = [300, 17, 250, 1]
    rows, heads, kv_heads, dim = sum(lengths), 12, 2, 128
    groups = heads // kv_heads
    q = torch.randn(1, rows, heads, dim, device="cuda", requires_grad=True)
    k = torch.randn(1, rows, kv_heads, dim, device="cuda", requires_grad=True)
    v = torch.randn(1, rows, kv_heads, dim, device="cuda", requires_grad=True)
    grad = torch.randn(1, rows, heads, dim, device="cuda").half()
    scale = dim ** -0.5

    pieces, start = [], 0
    qt = q.half().transpose(1, 2)
    kt = k.half().transpose(1, 2).repeat_interleave(groups, dim=1)
    vt = v.half().transpose(1, 2).repeat_interleave(groups, dim=1)
    for length in lengths:
        stop = start + length
        pieces.append(F.scaled_dot_product_attention(qt[:, :, start:stop], kt[:, :, start:stop],
                                                     vt[:, :, start:stop], scale=scale,
                                                     is_causal=True))
        start = stop
    ref = torch.cat(pieces, dim=2).transpose(1, 2)
    ref.backward(grad)
    ref_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None

    cu = torch.tensor([0, 300, 317, 567, 568], device="cuda", dtype=torch.int32)
    out = varlen_attention(q, k, v, cu, max(lengths), scale, groups)
    out.backward(grad)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)
    for got, want in zip((q.grad, k.grad, v.grad), ref_grads):
        rel = (got - want).norm() / want.norm()
        assert rel < 5e-3, f"gradient off by {rel:.2e} (relative norm)"
