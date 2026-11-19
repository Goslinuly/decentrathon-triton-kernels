"""The packed MLP path against the plain padded one: same values on kept rows, zeros on pad
rows, same gradients. Run on a GPU: `python -m pytest solution -q`."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("batch,seq,pad_tail", [(4, 96, (0, 31, 50, 96)), (2, 33, (5, 0))])
def test_packed_mlp_matches_padded(batch, seq, pad_tail):
    from transformers.models.qwen2 import modeling_qwen2

    from solution import unpad

    config = modeling_qwen2.Qwen2Config(hidden_size=64, intermediate_size=128, hidden_act="silu")
    torch.manual_seed(0)
    mlp = modeling_qwen2.Qwen2MLP(config).cuda().half()
    x = torch.randn(batch, seq, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    mask = torch.ones(batch, seq, device="cuda", dtype=torch.long)
    for row, tail in enumerate(pad_tail):
        if tail:
            mask[row, seq - tail:] = 0

    ref = modeling_qwen2.Qwen2MLP.forward(mlp, x)  # whatever forward is installed, unpacked
    grad = torch.randn_like(ref)
    ref.backward(grad)
    ref_grad, x.grad = x.grad.clone(), None

    keep = mask.reshape(-1).bool()
    unpad._state.mask, unpad._state.rows = mask, None   # as the model forward leaves them
    try:
        unpad.patch_unpad()  # idempotent for this test: wraps the current forward once more
        out = mlp(x)
    finally:
        unpad._state.mask = unpad._state.rows = None

    tol = dict(rtol=2e-3, atol=2e-3)
    flat_ref, flat_out = ref.reshape(-1, 64), out.reshape(-1, 64)
    torch.testing.assert_close(flat_out[keep], flat_ref[keep], **tol)
    assert (flat_out[~keep] == 0).all(), "pad rows must be zero"
    out.backward(grad)
    masked_ref = ref_grad.reshape(-1, 64).clone()
    masked_ref[~keep] = 0
    torch.testing.assert_close(x.grad.reshape(-1, 64), masked_ref, **tol)
