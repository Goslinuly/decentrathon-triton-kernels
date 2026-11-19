"""Causal attention over all the packed sequences of a batch in one call.

The packed layers ran SDPA once per sequence on slices of the packed q/k/v. In the backward
every slice becomes "zeros of the full tensor + copy the slice in", and the four slices' gradients
are then summed over the full tensor: per layer ~12 fills, ~12 copies and ~9 adds of
[1, heads, rows, 128] tensors, plus the transposes in and out.

PyTorch's memory-efficient attention (the same CUTLASS sm75 kernels SDPA picks on the T4) takes
variable-length batches natively: q/k/v as one [1, rows, heads, dim] tensor and `cu_seqlens`, the
start of each sequence; mask type 1 is causal from the top-left, i.e. inside each sequence. One
forward and one backward call per layer, no slices, no transposes. The values are SDPA's: the
same kernel, the same causal keys per sequence, q/k/v in fp16 as autocast feeds SDPA.

The kernel wants k and v with as many heads as q, so for grouped-query attention (12 query heads
over 2 kv heads here) they are repeated 6x. What the backward keeps are the 2-head k and v, and
the repeat is redone there: 2 x 5/6 of a [rows, 12, 128] fp16 tensor less per layer kept alive
across the step (about 215 MB at the peak), for one extra copy per tensor in the backward. The
gradients of the repeated tensors are summed over the repeats, exactly what autograd did for the
expand.

Plain PyTorch operators (`aten::_efficient_attention_forward/backward`), our own wrapper.
"""

import torch

_ops = getattr(torch.ops.aten, "_efficient_attention_forward", None), \
    getattr(torch.ops.aten, "_efficient_attention_backward", None)


def varlen_ok(query, plan):
    return (all(op is not None for op in _ops) and query.is_cuda
            and getattr(plan, "cu_seqlens", None) is not None and query.shape[-1] % 8 == 0
            and query.shape[-1] <= 128)


def _repeat_heads(t, groups):
    """[1, rows, kv_heads, dim] -> [1, rows, kv_heads * groups, dim]: head h reads kv head
    h // groups (as `repeat_kv`), materialized contiguous for the attention kernel."""
    if groups == 1:
        return t
    _, rows, kv_heads, dim = t.shape
    return t[:, :, :, None, :].expand(1, rows, kv_heads, groups, dim).reshape(1, rows, -1, dim)


def _sum_heads(grad, kv_heads, groups):
    """The gradient of `_repeat_heads`: summed over the repeats."""
    if groups == 1:
        return grad
    _, rows, _, dim = grad.shape
    return grad.view(1, rows, kv_heads, groups, dim).sum(dim=3)


class VarlenAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_len, scale, groups=1):
        out, lse, seed, offset, _, _ = _ops[0](q, _repeat_heads(k, groups), _repeat_heads(v, groups),
                                               None, cu_seqlens, cu_seqlens, max_len, max_len,
                                               0.0, 1, True, scale=scale)
        ctx.save_for_backward(q, k, v, out, lse, seed, offset, cu_seqlens)   # k, v: kv heads only
        ctx.max_len, ctx.scale, ctx.groups = max_len, scale, groups
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lse, seed, offset, cu_seqlens = ctx.saved_tensors
        groups, kv_heads = ctx.groups, k.shape[2]
        dq, dk, dv, _ = _ops[1](grad.contiguous(), q, _repeat_heads(k, groups),
                                _repeat_heads(v, groups), None, out, cu_seqlens, cu_seqlens,
                                ctx.max_len, ctx.max_len, lse, 0.0, seed, offset, 1, False,
                                scale=ctx.scale)
        return (dq, _sum_heads(dk, kv_heads, groups), _sum_heads(dv, kv_heads, groups),
                None, None, None, None)


def varlen_attention(query, key, value, cu_seqlens, max_len, scale, groups=1):
    """query [1, rows, heads, dim], key/value [1, rows, kv_heads, dim] -> [1, rows, heads, dim]."""
    q = query.to(torch.float16).contiguous()
    k, v = key.to(torch.float16).contiguous(), value.to(torch.float16).contiguous()
    return VarlenAttention.apply(q, k, v, cu_seqlens, max_len, scale, groups)
