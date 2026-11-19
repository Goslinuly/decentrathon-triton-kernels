"""Packed LoRA dropout: the same mask on the real rows as nn.Dropout on the padded grid."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("n_cols", [1536, 1000])
def test_packed_dropout_matches_nn_dropout(n_cols):
    from solution.lora_dropout import packed_dropout

    batch, seq = 4, 300
    keep = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
    for b, length in enumerate((300, 17, 250, 120)):
        keep[b, :length] = True
    rows = keep.reshape(-1).nonzero().squeeze(1)
    plan = SimpleNamespace(batch=batch, seq=seq, rows=rows)
    x = torch.randn(rows.numel(), n_cols, device="cuda", requires_grad=True)
    grad = torch.randn(rows.numel(), n_cols, device="cuda").half()
    dropout = torch.nn.Dropout(0.05).train()

    torch.manual_seed(7)
    grid = torch.zeros(batch * seq, n_cols, device="cuda").index_copy(0, rows, x)
    ref = dropout(grid).index_select(0, rows)
    ref.backward(grad.float())
    ref_dx, x.grad = x.grad.clone(), None
    after_ref = torch.cuda.get_rng_state()

    torch.manual_seed(7)
    out = packed_dropout(dropout, x, plan)
    out.backward(grad)
    assert torch.equal(torch.cuda.get_rng_state(), after_ref), "the RNG must advance as in the base"
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.half().float(), rtol=1e-3, atol=1e-3)
    assert torch.equal(out != 0, ref != 0), "kept positions must match the base mask"
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-5, atol=1e-5)


def test_packed_lora_matches_separate_path():
    """PackedLoRA (x16 kept once, dropped input rebuilt from the bits) against the separate
    path `lora_b(lora_a(packed_dropout(x))) * scaling` with the same mask: same output, same
    gradients for x and lora_B, lora_A's within one fp16 rounding of the input."""
    from types import SimpleNamespace

    from solution.lora_dropout import packed_dropout, packed_lora

    torch.manual_seed(0)
    batch, seq, cols, rank, out_features = 2, 40, 1536, 16, 256
    mask2d = torch.ones(batch, seq, device="cuda", dtype=torch.long)
    mask2d[1, 25:] = 0
    keep = mask2d.reshape(-1).bool()
    plan = SimpleNamespace(batch=batch, seq=seq, rows=keep.nonzero().squeeze(1))
    n = int(keep.sum())
    x = torch.randn(n, cols, device="cuda", requires_grad=True)
    dropout = torch.nn.Dropout(0.05).cuda().train()
    lora_a = torch.nn.Linear(cols, rank, bias=False).cuda()
    lora_b = torch.nn.Linear(rank, out_features, bias=False).cuda()
    torch.nn.init.normal_(lora_b.weight, std=0.1)
    scaling = 2.0

    with torch.autocast("cuda", dtype=torch.float16):
        torch.manual_seed(7)
        ref = lora_b(lora_a(packed_dropout(dropout, x, plan))) * scaling
    ref.float().sum().backward()
    ref_grads = [x.grad.clone(), lora_a.weight.grad.clone(), lora_b.weight.grad.clone()]
    x.grad = lora_a.weight.grad = lora_b.weight.grad = None

    with torch.autocast("cuda", dtype=torch.float16):
        torch.manual_seed(7)
        out = packed_lora(dropout, x, x.detach().to(torch.float16), plan, lora_a, lora_b, scaling)
    assert out is not None and out.dtype == torch.float16
    out.float().sum().backward()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x.grad, ref_grads[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lora_b.weight.grad, ref_grads[2], rtol=1e-3, atol=1e-3)
    rel = (lora_a.weight.grad - ref_grads[1]).norm() / ref_grads[1].norm()
    assert rel < 5e-3, f"lora_A gradient off by {rel:.2e} (relative norm)"
