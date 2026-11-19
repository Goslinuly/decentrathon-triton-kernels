"""Output head and causal-LM loss without the [tokens, vocab] logits, with a Triton row kernel.

Base Soup runs `lm_head` over every position (up to 2,048 x 151,936 logits), upcasts them to fp32
for the loss, keeps the softmax for the backward, and reads them again for TRL's entropy and token
accuracy and for Soup's per-row rewind losses. Here:

- only positions with attention go through the head, in chunks of `CHUNK` rows, the labelled
  ones first; each chunk's logits are one fp16 cuBLAS matmul (as the base's fp16 autocast
  matmul) into a reused buffer;
- `head_ce_rows_kernel` (ours) reads each row of the chunk once for an online softmax: log-sum-
  exp, entropy, argmax and the label's logit; for labelled rows it reads the row again and
  writes the gradient (softmax - onehot) * GRAD_SCALE / n over the logits in place;
- the gradient for the hidden states, dlogits @ W, is one more cuBLAS matmul per chunk, done in
  the forward, so the backward only scales it: no logits outlive their chunk and the head's
  matmul is never recomputed.

Triton 3.6 does not use the T4's tensor cores for `tl.dot` (sm_75), so the matmuls stay in
cuBLAS and the kernel does everything around them.

Ideas, with our own code: Liger-Kernel's fused linear cross-entropy (chunking over rows, the
gradient written over the logits in the forward), https://github.com/linkedin/Liger-Kernel;
Cut Cross-Entropy (Wijmans et al. 2024, arXiv:2411.09009); online softmax (Milakov &
Gimelshein 2018, arXiv:1805.02867).
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK = 128            # rows per chunk: the fp16 logits buffer is 128 x 151,936 x 2 B = 39 MB
BLOCK_V = 4096
# The fp16 gradient is (softmax - onehot) * GRAD_SCALE / n. The base rounds the same quantity
# times the AMP loss scale (2**16 at the start) to fp16; a fixed 2**15 keeps |g| <= 32768 for
# any n >= 1 and keeps the small terms out of fp16's subnormal range. The backward divides it out.
GRAD_SCALE = 2.0 ** 15


# `n_grad_rows` differs from chunk to chunk; without `do_not_specialize` Triton compiles one more
# variant whenever its divisibility by 16 changes, in the middle of the timed steps.
@triton.jit(do_not_specialize=["n_grad_rows"])
def head_ce_rows_kernel(logits_ptr, tgt_ptr, lse_ptr, ent_ptr, amax_ptr, tz_ptr,
                        n_grad_rows, vocab, stride, grad_mult,
                        BLOCK_V: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride
    tgt = tl.load(tgt_ptr + row)
    offs = tl.arange(0, BLOCK_V)

    run_max = float("-inf")
    sum_exp = 0.0
    sum_exp_z = 0.0
    best = float("-inf")
    best_idx = 0
    tgt_z = 0.0
    for v0 in range(0, vocab, BLOCK_V):
        cols = v0 + offs
        ok = cols < vocab
        z = tl.load(base + cols, mask=ok, other=float("-inf")).to(tl.float32)
        tile_max = tl.max(z, axis=0)
        new_max = tl.maximum(run_max, tile_max)
        alpha = tl.exp(run_max - new_max)
        e = tl.exp(z - new_max)
        sum_exp = sum_exp * alpha + tl.sum(e, axis=0)
        sum_exp_z = sum_exp_z * alpha + tl.sum(tl.where(ok, e * z, 0.0), axis=0)
        run_max = new_max
        tile_arg = tl.argmax(z, axis=0) + v0
        better = tile_max > best        # strict: the first index wins a tie, as torch.argmax
        best_idx = tl.where(better, tile_arg, best_idx)
        best = tl.where(better, tile_max, best)
        tgt_z += tl.sum(tl.where(cols == tgt, z, 0.0), axis=0)

    lse = run_max + tl.log(sum_exp)
    tl.store(lse_ptr + row, lse)
    tl.store(ent_ptr + row, lse - sum_exp_z / sum_exp)
    tl.store(amax_ptr + row, best_idx)
    tl.store(tz_ptr + row, tgt_z)

    if row < n_grad_rows:               # labelled rows come first in every chunk
        for v0 in range(0, vocab, BLOCK_V):
            cols = v0 + offs
            ok = cols < vocab
            z = tl.load(base + cols, mask=ok, other=0.0).to(tl.float32)
            g = (tl.exp(z - lse) - tl.where(cols == tgt, 1.0, 0.0)) * grad_mult
            tl.store(base + cols, g.to(logits_ptr.dtype.element_ty), mask=ok)


_w16_cache = {}


def fp16_weight(weight):
    """The head's weight in fp16 for cuBLAS. It is the tied embedding, frozen during LoRA
    training, so one copy serves every step; a changed weight (another version) is cast again."""
    if weight.dtype == torch.float16:
        return weight
    key = (weight.data_ptr(), weight._version, tuple(weight.shape))
    cached = _w16_cache.get("w")
    if cached is None or cached[0] != key:
        _w16_cache["w"] = (key, weight.detach().to(torch.float16).contiguous())
    return _w16_cache["w"][1]


class FusedHeadLoss(torch.autograd.Function):
    """Sum of the token cross-entropies of the labelled rows / denominator, and the metrics.

    `hidden` [rows, dim] are the head's inputs with the `n_labelled` labelled rows first,
    `targets` [rows] the next-token labels (-100 = not scored). The weight gets no gradient:
    it is the frozen tied embedding.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, n_labelled, denominator):
        w16 = fp16_weight(weight.detach())
        h16 = hidden.detach().to(torch.float16).contiguous()
        rows, dim = h16.shape
        vocab = w16.shape[0]
        want_grad = ctx.needs_input_grad[0]
        dev = h16.device
        lse = torch.empty(rows, device=dev, dtype=torch.float32)
        entropy = torch.empty_like(lse)
        tgt_z = torch.empty_like(lse)
        argmax = torch.empty(rows, device=dev, dtype=torch.int32)
        dx = torch.empty(n_labelled if want_grad else 0, dim, device=dev, dtype=torch.float16)
        grad_mult = GRAD_SCALE / float(denominator)
        buf = torch.empty(min(CHUNK, rows), vocab, device=dev, dtype=torch.float16)
        for start in range(0, rows, CHUNK):
            stop = min(start + CHUNK, rows)
            logits = buf[: stop - start]
            torch.mm(h16[start:stop], w16.t(), out=logits)
            n_grad = max(0, min(n_labelled, stop) - start) if want_grad else 0
            head_ce_rows_kernel[(stop - start,)](
                logits, targets[start:stop], lse[start:stop], entropy[start:stop],
                argmax[start:stop], tgt_z[start:stop], n_grad, vocab, logits.stride(0),
                grad_mult, BLOCK_V=BLOCK_V, num_warps=8)
            if n_grad:
                torch.mm(logits[:n_grad], w16, out=dx[start:start + n_grad])
        del buf
        labelled = targets != IGNORE_INDEX
        token_loss = torch.where(labelled, lse - tgt_z, torch.zeros_like(lse))
        loss = token_loss.sum() / denominator
        ctx.save_for_backward(dx)
        ctx.rows, ctx.hidden_dtype = rows, hidden.dtype
        ctx.mark_non_differentiable(token_loss, entropy, argmax)
        return loss, token_loss, entropy, argmax

    @staticmethod
    def backward(ctx, grad_loss, _grad_token_loss, _grad_entropy, _grad_argmax):
        (dx,) = ctx.saved_tensors
        grad = torch.zeros(ctx.rows, dx.shape[1], device=dx.device, dtype=ctx.hidden_dtype)
        grad[: dx.shape[0]] = dx.to(grad.dtype) * (grad_loss.to(grad.dtype) / GRAD_SCALE)
        return grad, None, None, None, None


@dataclass(frozen=True)
class HeadStep:
    """What one labelled forward leaves for TRL's metrics and Soup's rewind log, in the shape
    `soup_cli.utils.stream_loss_chunk.ChunkedStep` has (Soup's recorder reads `token_loss`)."""

    token_entropy: torch.Tensor   # [batch, seq]
    token_loss: torch.Tensor      # [batch, seq] fp32: CE of labels[:, t + 1] at t, else 0
    correct: torch.Tensor         # labelled positions whose argmax is the label
    labelled: torch.Tensor        # labelled positions


def fused_head_loss(hidden, weight, labels, attention_mask=None, num_items_in_batch=None):
    """`ForCausalLMLoss(lm_head(hidden), labels)` and the per-token metrics, without logits.

    Positions without attention are skipped: their loss and entropy are 0, and TRL weighs the
    entropy by the attention mask, so they never count.
    """
    batch, seq, dim = hidden.shape
    targets = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
    targets = targets.to(hidden.device).reshape(-1)
    if attention_mask is not None:
        keep = attention_mask.reshape(-1).to(hidden.device).bool()
        targets = torch.where(keep, targets, torch.full_like(targets, IGNORE_INDEX))
    else:
        keep = torch.ones_like(targets, dtype=torch.bool)
    labelled = targets != IGNORE_INDEX
    rows = torch.cat([labelled.nonzero().squeeze(1), (keep & ~labelled).nonzero().squeeze(1)])
    n_labelled = int(labelled.sum())

    if num_items_in_batch is None:
        denominator = labelled.sum()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch

    picked = hidden.reshape(batch * seq, dim).index_select(0, rows)
    row_targets = targets.index_select(0, rows).contiguous()
    loss, token_loss, entropy, argmax = FusedHeadLoss.apply(picked, weight, row_targets,
                                                           n_labelled, denominator)

    def scatter(values):
        full = torch.zeros(batch * seq, device=hidden.device, dtype=torch.float32)
        full.index_copy_(0, rows, values.float())
        return full.view(batch, seq)

    step = HeadStep(
        token_entropy=scatter(entropy),
        token_loss=scatter(token_loss),
        correct=(argmax[:n_labelled].long() == row_targets[:n_labelled]).sum(),
        labelled=labelled.sum(),
    )
    return loss, step
