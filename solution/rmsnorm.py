"""Qwen2RMSNorm forward and backward in one Triton launch each.

After `prepare_model_for_kbit_training` the norm weights are fp32 and the residual stream is
fp32, so each of the 57 norms of a step is ~6 PyTorch kernels over a [tokens, 1536] fp32 tensor
(cast, pow, mean, add + rsqrt, mul, mul by the weight), and autograd keeps several of their
fp32 results for the backward. Here one program normalizes one row; the backward needs only the
input and one fp32 `rstd` per row. The weight is frozen in LoRA training, so it gets no
gradient; a trainable weight takes the original forward.

Idea and math: Zhang & Sennrich, Root Mean Square Layer Normalization (2019,
arXiv:1910.07467); the row-per-program layout of Triton's layer-norm tutorial
(https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html). Our own code.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, xs_ptr, rstd_ptr, n_cols, eps, stride_x, stride_y,
                       ROUND_TO_INPUT: tl.constexpr, SAVE_HALF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    x = tl.load(x_ptr + row.to(tl.int64) * stride_x + cols, mask=ok, other=0.0)
    xf = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / n_cols + eps)
    normed = xf * rstd
    if ROUND_TO_INPUT:                  # as `hidden_states.to(input_dtype)` in the original
        normed = normed.to(x.dtype).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
    tl.store(y_ptr + row.to(tl.int64) * stride_y + cols,
             (w * normed).to(y_ptr.dtype.element_ty), mask=ok)
    tl.store(rstd_ptr + row, rstd)
    if SAVE_HALF:                       # the backward's copy of the input, rows packed
        tl.store(xs_ptr + row.to(tl.int64) * n_cols + cols, x.to(tl.float16), mask=ok)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols,
                       stride_dy, stride_x, stride_dx, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    dy = tl.load(dy_ptr + row.to(tl.int64) * stride_dy + cols, mask=ok, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row.to(tl.int64) * stride_x + cols, mask=ok, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    wdy = w * dy
    # d/dx of w * x * rstd(x): rstd * (w*dy - xhat * mean(w*dy * xhat))
    dx = rstd * (wdy - xhat * (tl.sum(wdy * xhat, axis=0) / n_cols))
    tl.store(dx_ptr + row.to(tl.int64) * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty),
             mask=ok)


class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        rows, cols = x2.shape
        out_dtype = torch.promote_types(weight.dtype, x.dtype)
        y = torch.empty(rows, cols, device=x.device, dtype=out_dtype)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(cols)
        # An fp32 input is kept for the backward as fp16: half the bytes of the largest thing a
        # norm keeps; rstd stays exact (fp32, from the fp32 input).
        save_half = x.dtype == torch.float32
        xs = torch.empty(rows, cols, device=x.device, dtype=torch.float16) if save_half else x2
        rmsnorm_fwd_kernel[(rows,)](x2, weight, y, xs, rstd, cols, eps, x2.stride(0), y.stride(0),
                                    ROUND_TO_INPUT=x.dtype != torch.float32, SAVE_HALF=save_half,
                                    BLOCK=block, num_warps=8 if block >= 2048 else 4)
        ctx.save_for_backward(xs, weight, rstd)
        ctx.shape, ctx.x_dtype = shape, x.dtype
        return y.view(*shape[:-1], cols)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, cols = x2.shape
        dy2 = dy.reshape(rows, cols)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty(rows, cols, device=x2.device, dtype=ctx.x_dtype)
        block = triton.next_power_of_2(cols)
        rmsnorm_bwd_kernel[(rows,)](dy2, x2, weight, rstd, dx, cols,
                                    dy2.stride(0), x2.stride(0), dx.stride(0), BLOCK=block,
                                    num_warps=8 if block >= 2048 else 4)
        return dx.view(ctx.shape), None, None


def rmsnorm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    original = cls.forward

    def forward(self, hidden_states):
        w = self.weight
        if (not hidden_states.is_cuda or w.requires_grad or hidden_states.shape[-1] > 8192
                or hidden_states.dtype not in (torch.float32, torch.float16)):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, w, float(self.variance_epsilon))

    cls.forward = forward
