"""The q/v LoRA dropout on the packed rows, with the base's masks, kept as bits.

The base draws each LoRA dropout mask over the padded [batch * seq, 1536] fp32 input. To draw
the same masks, the packed layers used to scatter the rows back into a zero grid, run
`nn.Dropout` on it and gather the rows again; the backward did the same in reverse, and
autograd kept a padded bool mask and an fp16 copy of the dropped input per projection.

Here the mask comes from `torch.ops.aten.native_dropout` on an uninitialized fp32 tensor of the
padded shape: `F.dropout` on a CUDA tensor is exactly that call, so it consumes the RNG the
same way and the mask on every real row is the base's, bit for bit (a dropout mask does not
depend on the values). Then our kernels work on the real rows only:

- `lora_dropout_fwd_kernel`: dropped = x * (1 / (1 - p)) where kept, in fp32 as the base
  rounds it, written as fp16 (what autocast feeds `lora_A`), and the row's mask packed to bits;
- `lora_dropout_bwd_kernel`: d_x = d_dropped * mask * (1 / (1 - p)) from the bits; the same
  kernel re-applies a kept mask to an fp16 input.

`PackedLoRA` is the whole branch `lora_B(lora_A(dropout(x))) * scaling` as one autograd node
with the arithmetic autocast would do (fp16 matmuls, fp32 parameter gradients). What it keeps
for the backward is the layer's fp16 input once, shared by q_proj and v_proj and with the 4-bit
projections, plus the mask bits and the rank-16 activations: the two fp16 copies of the dropped
input that `lora_A` kept per layer (one per projection) are gone, 3 KB per token per layer. In
the backward the dropped input is rebuilt from that fp16 copy and the bits; against the base
that is one fp16 rounding of the input before the scale instead of after it, a difference of
at most one fp16 ulp on some elements of `lora_A`'s weight gradient only.

Saved per projection and token: 192 bytes of mask instead of a padded 1,536-byte mask.
Our own kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK = 2048


@triton.jit(do_not_specialize=["n_rows"])
def lora_dropout_fwd_kernel(x_ptr, mask_ptr, rows_ptr, out_ptr, bits_ptr, n_rows, n_cols, scale,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    src = tl.load(rows_ptr + row)                    # this row's place in the padded grid
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    x = tl.load(x_ptr + row.to(tl.int64) * n_cols + cols, mask=ok, other=0.0).to(tl.float32)
    keep = tl.load(mask_ptr + src * n_cols + cols, mask=ok, other=0).to(tl.int32)
    out = tl.where(keep != 0, x * scale, 0.0)
    tl.store(out_ptr + row.to(tl.int64) * n_cols + cols, out.to(tl.float16), mask=ok)
    weights = (1 << tl.arange(0, 8)).to(tl.int32)
    packed = tl.sum(tl.reshape(keep, (BLOCK // 8, 8)) * weights[None, :], axis=1)
    bytes_ = tl.arange(0, BLOCK // 8)
    tl.store(bits_ptr + row.to(tl.int64) * (BLOCK // 8) + bytes_, packed.to(tl.uint8))


@triton.jit(do_not_specialize=["n_rows"])
def lora_dropout_bwd_kernel(g_ptr, bits_ptr, dx_ptr, n_rows, n_cols, scale, BLOCK: tl.constexpr):
    # dx = g * scale where the bit is set, in the dtype of `dx_ptr` (fp32 for a gradient,
    # fp16 when re-applying the mask to the saved fp16 input).
    row = tl.program_id(0)
    bytes_ = tl.load(bits_ptr + row.to(tl.int64) * (BLOCK // 8) + tl.arange(0, BLOCK // 8)).to(tl.int32)
    keep = tl.reshape((bytes_[:, None] >> tl.arange(0, 8)[None, :]) & 1, (BLOCK,))
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    g = tl.load(g_ptr + row.to(tl.int64) * n_cols + cols, mask=ok, other=0.0).to(tl.float32)
    tl.store(dx_ptr + row.to(tl.int64) * n_cols + cols,
             tl.where(keep != 0, g * scale, 0.0).to(dx_ptr.dtype.element_ty), mask=ok)


def _apply_bits(x, bits, scale, dtype):
    """x * scale where the bit is set, else 0, written in `dtype`."""
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, n_cols, device=x.device, dtype=dtype)
    lora_dropout_bwd_kernel[(n_rows,)](x.contiguous(), bits, out, n_rows, n_cols, scale,
                                       BLOCK=BLOCK, num_warps=4)
    return out


class PackedDropout(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask, rows, scale):
        n_rows, n_cols = x.shape
        x = x.contiguous()
        out = torch.empty(n_rows, n_cols, device=x.device, dtype=torch.float16)
        bits = torch.empty(n_rows, BLOCK // 8, device=x.device, dtype=torch.uint8)
        lora_dropout_fwd_kernel[(n_rows,)](x, mask.view(torch.uint8), rows, out, bits, n_rows,
                                           n_cols, scale, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(bits)
        ctx.scale, ctx.n_cols, ctx.x_dtype = scale, n_cols, x.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        (bits,) = ctx.saved_tensors
        dx = _apply_bits(grad, bits, ctx.scale, torch.float32)
        return dx.to(ctx.x_dtype), None, None, None


def _mask(dropout, plan, n_cols, device, dtype):
    """The base's dropout mask: the same `native_dropout` call over the same padded shape."""
    grid = torch.empty(plan.batch * plan.seq, n_cols, device=device, dtype=dtype)
    _, mask = torch.ops.aten.native_dropout(grid, dropout.p, True)
    return mask


def _plain_dropout(dropout, x):
    return (type(dropout) is torch.nn.Dropout and dropout.training and x.is_cuda
            and x.dtype == torch.float32 and x.shape[-1] <= BLOCK and 0.0 < dropout.p < 1.0)


def packed_dropout(dropout, x, plan):
    """`dropout(grid of x)` gathered back to the packed rows, as fp16, or None if this dropout
    is not the plain training-mode `nn.Dropout` on an fp32 input of at most BLOCK features."""
    if not _plain_dropout(dropout, x):
        return None
    mask = _mask(dropout, plan, x.shape[-1], x.device, x.dtype)
    return PackedDropout.apply(x, mask, plan.rows, 1.0 / (1.0 - dropout.p))


class PackedLoRA(torch.autograd.Function):
    """lora_B(lora_A(dropout(x))) * scaling on the packed rows; keeps x16, the bits and the
    rank-r activations for the backward."""

    @staticmethod
    def forward(ctx, x, x16, mask, rows, weight_a, weight_b, scale, scaling):
        n_rows, n_cols = x.shape
        x = x.contiguous()
        dropped = torch.empty(n_rows, n_cols, device=x.device, dtype=torch.float16)
        bits = torch.empty(n_rows, BLOCK // 8, device=x.device, dtype=torch.uint8)
        lora_dropout_fwd_kernel[(n_rows,)](x, mask.view(torch.uint8), rows, dropped, bits, n_rows,
                                           n_cols, scale, BLOCK=BLOCK, num_warps=4)
        a16, b16 = weight_a.to(torch.float16), weight_b.to(torch.float16)   # as autocast casts them
        h = F.linear(dropped, a16)                     # [rows, r] fp16
        out = F.linear(h, b16) * scaling               # [rows, out] fp16
        ctx.save_for_backward(x16, bits, h, weight_a, weight_b)
        ctx.scale, ctx.scaling, ctx.x_dtype = scale, scaling, x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x16, bits, h, weight_a, weight_b = ctx.saved_tensors
        a16, b16 = weight_a.to(torch.float16), weight_b.to(torch.float16)
        g = grad_out.to(torch.float16).contiguous() * ctx.scaling   # d(out before scaling)
        grad_b = (g.t() @ h).to(weight_b.dtype)                     # [out, r]
        dh = g @ b16                                                # [rows, r] fp16
        dropped = _apply_bits(x16, bits, ctx.scale, torch.float16)  # lora_A's input, rebuilt
        grad_a = (dh.t() @ dropped).to(weight_a.dtype)              # [r, cols]
        d_dropped = dh @ a16                                        # [rows, cols] fp16
        dx = _apply_bits(d_dropped, bits, ctx.scale, torch.float32).to(ctx.x_dtype)
        return dx, None, None, None, grad_a, grad_b, None, None


def _plain_linear(module):
    return (type(module) is torch.nn.Linear and module.bias is None
            and module.weight.is_cuda and module.weight.dtype == torch.float32)


def packed_lora(dropout, x, x16, plan, lora_a, lora_b, scaling):
    """The LoRA branch on the packed rows with the base's dropout mask, keeping `x16` (the
    layer's fp16 input, shared by the projections) instead of a dropped copy per projection;
    None when the branch is not the plain `Dropout -> Linear -> Linear` of peft's LoRA."""
    if (not _plain_dropout(dropout, x) or not _plain_linear(lora_a) or not _plain_linear(lora_b)
            or not torch.is_tensor(x16) or x16.dtype != torch.float16 or x16.shape != x.shape
            or not torch.is_autocast_enabled()):
        return None
    mask = _mask(dropout, plan, x.shape[-1], x.device, x.dtype)
    return PackedLoRA.apply(x, x16, mask, plan.rows, lora_a.weight, lora_b.weight,
                            1.0 / (1.0 - dropout.p), float(scaling))
