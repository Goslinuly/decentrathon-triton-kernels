"""The fused head against plain PyTorch: loss, entropy, accuracy, per-token losses, gradient.

The reference does what the base run does under fp16 autocast: an fp16 matmul, the logits
upcast to fp32, `cross_entropy`. Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(hidden, weight, labels, attention_mask):
    logits = (hidden.half() @ weight.half().T).float()
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    flat = logits.view(-1, logits.shape[-1])
    token_loss = torch.nn.functional.cross_entropy(flat, targets.reshape(-1), reduction="none",
                                                   ignore_index=-100).view(targets.shape)
    labelled = targets != -100
    loss = token_loss.sum() / labelled.sum()
    logp = torch.log_softmax(logits, -1)
    entropy = -(logp.exp() * logp).sum(-1)
    correct = ((logits.argmax(-1) == targets) & labelled).sum()
    return loss, token_loss, entropy * attention_mask, correct, labelled.sum()


@pytest.mark.parametrize("shape", [(4, 96, 1536, 151936), (3, 77, 96, 1001), (2, 33, 64, 300)])
def test_fused_head_matches_pytorch(shape):
    from solution.fused_head import fused_head_loss

    batch, seq, dim, vocab = shape
    torch.manual_seed(0)
    hidden = (torch.randn(batch, seq, dim, device="cuda") * 0.5).requires_grad_(True)
    weight = torch.randn(vocab, dim, device="cuda") * dim ** -0.5
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    attention_mask = torch.ones(batch, seq, device="cuda", dtype=torch.long)
    attention_mask[0, seq // 2:] = 0                      # a padded row
    labels[attention_mask == 0] = -100
    labels[:, : seq // 4] = -100                          # an unscored prompt

    ref_loss, ref_token_loss, ref_entropy, ref_correct, ref_labelled = reference(
        hidden, weight, labels, attention_mask)
    ref_loss.backward()
    ref_grad, hidden.grad = hidden.grad.clone(), None

    loss, step = fused_head_loss(hidden, weight, labels, attention_mask)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(step.token_loss, ref_token_loss, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(step.token_entropy, ref_entropy, rtol=2e-3, atol=2e-3)
    assert abs(int(step.correct) - int(ref_correct)) <= max(1, int(ref_labelled) // 200)
    assert int(step.labelled) == int(ref_labelled)
    rel = (hidden.grad - ref_grad).norm() / ref_grad.norm()
    assert rel < 1e-2, f"gradient off by {rel:.2e} (relative norm)"
