"""SwiGLU for the Qwen2 MLP that keeps its saved activations in int8.

The MLP's backward needs `gate` and `up` ([tokens, 8960] each): with the 4-bit projections
frozen, they are the largest thing a training step keeps alive between the forward and the
backward, 35.8 KB per token per layer in fp16, over half of all saved activations. The task
statement lists activation compression among the directions; this is it for these two tensors.

- `swiglu8_fwd_kernel`: out = silu(gate) * up in fp32, rounded to fp16 for `down_proj`; in the
  same pass `gate` and `up` are quantized to int8 with one fp32 scale per 64 values
  (absmax / 127, round to nearest). 18 KB per token per layer instead of 35.8 KB; the fp16
  `gate` and `up` are freed as soon as the forward returns.
- `swiglu8_bwd_kernel`: dequantizes `gate` and `up` in registers and writes
  d_gate = d_out * up * silu'(gate), d_up = d_out * silu(gate).

The forward's output is exact (computed from the fp16 inputs before quantization); only the
backward sees the 8-bit values: an error of at most absmax / 254 per element within each block
of 64, in a gradient that then goes through `gate_proj` / `up_proj` and the residual stream to
the q/v adapters. The harness checks bound the result (loss curve, per-row losses, eval loss).

Our own kernels (not the organizers' example): the blockwise absmax int8 scheme follows
8-bit blockwise quantization (Dettmers et al., 8-bit Optimizers via Block-wise Quantization,
arXiv:2110.02861) and activation compression for training (ActNN, Chen et al. 2021,
arXiv:2104.14129).
"""

import torch
import triton
import triton.language as tl

QBLOCK = 64          # values per int8 scale
ROWS_PER_PROG = 4    # [rows, QBLOCK * COLS_PER_PROG] tile per program
QBLOCKS_PER_PROG = 8


# n_rows changes with every batch; specializing on it would compile again mid-run.
@triton.jit(do_not_specialize=["n_rows"])
def swiglu8_fwd_kernel(gate_ptr, up_ptr, out_ptr, qg_ptr, qu_ptr, sg_ptr, su_ptr,
                       n_rows, n_cols, n_blocks,
                       ROWS: tl.constexpr, QBLOCK: tl.constexpr, NB: tl.constexpr):
    # One program: ROWS rows x NB quantization blocks of QBLOCK columns.
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    b = tl.program_id(1) * NB + tl.arange(0, NB)
    c = tl.arange(0, QBLOCK)
    row_ok = r < n_rows
    blk_ok = b < n_blocks
    cols = b[:, None] * QBLOCK + c[None, :]                       # [NB, QBLOCK]
    ok = row_ok[:, None, None] & blk_ok[None, :, None] & (cols[None, :, :] < n_cols)
    offs = r.to(tl.int64)[:, None, None] * n_cols + cols[None, :, :]
    g = tl.load(gate_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty), mask=ok)

    s_offs = r.to(tl.int64)[:, None] * n_blocks + b[None, :]
    s_ok = row_ok[:, None] & blk_ok[None, :]
    sg = tl.max(tl.abs(g), axis=2) / 127.0
    su = tl.max(tl.abs(u), axis=2) / 127.0
    tl.store(sg_ptr + s_offs, sg, mask=s_ok)
    tl.store(su_ptr + s_offs, su, mask=s_ok)
    qg = g / tl.where(sg == 0.0, 1.0, sg)[:, :, None]
    qu = u / tl.where(su == 0.0, 1.0, su)[:, :, None]
    qg = tl.where(qg >= 0, qg + 0.5, qg - 0.5).to(tl.int8)      # round half away from zero
    qu = tl.where(qu >= 0, qu + 0.5, qu - 0.5).to(tl.int8)
    tl.store(qg_ptr + offs, qg, mask=ok)
    tl.store(qu_ptr + offs, qu, mask=ok)


@triton.jit(do_not_specialize=["n_rows"])
def swiglu8_bwd_kernel(dout_ptr, qg_ptr, qu_ptr, sg_ptr, su_ptr, dg_ptr, du_ptr,
                       n_rows, n_cols, n_blocks,
                       ROWS: tl.constexpr, QBLOCK: tl.constexpr, NB: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    b = tl.program_id(1) * NB + tl.arange(0, NB)
    c = tl.arange(0, QBLOCK)
    row_ok = r < n_rows
    blk_ok = b < n_blocks
    cols = b[:, None] * QBLOCK + c[None, :]
    ok = row_ok[:, None, None] & blk_ok[None, :, None] & (cols[None, :, :] < n_cols)
    offs = r.to(tl.int64)[:, None, None] * n_cols + cols[None, :, :]
    s_offs = r.to(tl.int64)[:, None] * n_blocks + b[None, :]
    s_ok = row_ok[:, None] & blk_ok[None, :]
    sg = tl.load(sg_ptr + s_offs, mask=s_ok, other=0.0)
    su = tl.load(su_ptr + s_offs, mask=s_ok, other=0.0)
    g = tl.load(qg_ptr + offs, mask=ok, other=0).to(tl.float32) * sg[:, :, None]
    u = tl.load(qu_ptr + offs, mask=ok, other=0).to(tl.float32) * su[:, :, None]
    dout = tl.load(dout_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    tl.store(du_ptr + offs, (dout * silu).to(du_ptr.dtype.element_ty), mask=ok)
    tl.store(dg_ptr + offs, (dout * u * sig * (1.0 + g * (1.0 - sig))).to(dg_ptr.dtype.element_ty),
             mask=ok)


def _grid(rows, n_blocks):
    return (triton.cdiv(rows, ROWS_PER_PROG), triton.cdiv(n_blocks, QBLOCKS_PER_PROG))


class SwiGLU8(torch.autograd.Function):
    """silu(gate) * up; saves gate and up as blockwise int8 for the backward."""

    @staticmethod
    def forward(ctx, gate, up):
        shape = gate.shape
        n_cols = shape[-1]
        gate = gate.reshape(-1, n_cols).contiguous()
        up = up.to(gate.dtype).reshape(-1, n_cols).contiguous()
        rows = gate.shape[0]
        n_blocks = triton.cdiv(n_cols, QBLOCK)
        out = torch.empty_like(gate)
        qg = torch.empty(rows, n_cols, device=gate.device, dtype=torch.int8)
        qu = torch.empty_like(qg)
        sg = torch.empty(rows, n_blocks, device=gate.device, dtype=torch.float32)
        su = torch.empty_like(sg)
        swiglu8_fwd_kernel[_grid(rows, n_blocks)](
            gate, up, out, qg, qu, sg, su, rows, n_cols, n_blocks,
            ROWS=ROWS_PER_PROG, QBLOCK=QBLOCK, NB=QBLOCKS_PER_PROG, num_warps=4)
        ctx.save_for_backward(qg, qu, sg, su)
        ctx.shape, ctx.dtype = shape, gate.dtype
        return out.view(shape)

    @staticmethod
    def backward(ctx, dout):
        qg, qu, sg, su = ctx.saved_tensors
        rows, n_cols = qg.shape
        n_blocks = sg.shape[1]
        dout = dout.reshape(rows, n_cols).contiguous()
        dg = torch.empty(rows, n_cols, device=qg.device, dtype=ctx.dtype)
        du = torch.empty_like(dg)
        swiglu8_bwd_kernel[_grid(rows, n_blocks)](
            dout, qg, qu, sg, su, dg, du, rows, n_cols, n_blocks,
            ROWS=ROWS_PER_PROG, QBLOCK=QBLOCK, NB=QBLOCKS_PER_PROG, num_warps=4)
        return dg.view(ctx.shape), du.view(ctx.shape)


def swiglu8(gate, up):
    return SwiGLU8.apply(gate, up)


def patch_qwen2_mlp_int8():
    """Qwen2MLP.forward through `swiglu8`, in place of the organizers' example patch."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda or not torch.is_grad_enabled():
            return original(self, x)
        return self.down_proj(swiglu8(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
