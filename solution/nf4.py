"""NF4 dequantization of the 4-bit base weights in one Triton launch.

With 2,048 tokens per batch, bitsandbytes 0.50.2 trains through dequantize + `F.linear` and
dequantizes again in the backward, every one of the 7 linear layers of the 28 blocks. With
double quantization each dequantization is two kernels: the 8-bit block scales first
(`dequantize_blockwise`, written out to memory), then the 4-bit weights. Here one kernel
reads the packed weight, the 8-bit scale and its second-level scale, and writes the fp16
weight: value = NF4[q] * (code8[absmax8[block]] * absmax2[block // 256] + offset).

We replace only the two dequantization functions bitsandbytes calls on this path
(`functional.dequantize_4bit` and the CUDA backend's `_dequant_linear_fallback`); the
matmuls and bitsandbytes' own control flow stay as they are. Any other format (fp4, no
double quantization, other block sizes) goes to the original functions.

The layout (two values per byte, the first in the high nibble; blockwise absmax) follows
bitsandbytes' reference implementation in `backends/default/ops.py`. Our own kernel.
"""

import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax8_ptr, absmax2_ptr, code8_ptr, nf4_ptr, offset_ptr,
                       out_ptr, n, blocksize, blocksize2, BLOCK: tl.constexpr):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = idx < n
    byte = tl.load(packed_ptr + idx // 2, mask=ok, other=0).to(tl.int32)
    q = tl.where(idx % 2 == 0, byte >> 4, byte & 15)
    value = tl.load(nf4_ptr + q, mask=ok, other=0.0)
    block = idx // blocksize
    a8 = tl.load(absmax8_ptr + block, mask=ok, other=0).to(tl.int32)
    scale = (tl.load(code8_ptr + a8, mask=ok, other=0.0)
             * tl.load(absmax2_ptr + block // blocksize2, mask=ok, other=0.0)
             + tl.load(offset_ptr))
    tl.store(out_ptr + idx, (value * scale).to(out_ptr.dtype.element_ty), mask=ok)


_nf4_tables = {}


def _nf4_table(device):
    table = _nf4_tables.get(device)
    if table is None:
        from bitsandbytes.functional import get_4bit_type

        table = get_4bit_type("nf4", device=device).float().contiguous()
        _nf4_tables[device] = table
    return table


def nf4_dequant(packed, absmax8, absmax2, code8, offset, blocksize, blocksize2, shape, dtype):
    n = 1
    for d in shape:
        n *= d
    out = torch.empty(shape, device=packed.device, dtype=dtype)
    if not torch.is_tensor(offset):
        offset = torch.tensor(float(offset), device=packed.device)
    offset = offset.to(device=packed.device, dtype=torch.float32).reshape(1)
    nf4_dequant_kernel[(triton.cdiv(n, BLOCK),)](
        packed.reshape(-1).view(torch.uint8), absmax8.reshape(-1), absmax2.reshape(-1).float(),
        code8.float(), _nf4_table(packed.device), offset, out, n, blocksize, blocksize2,
        BLOCK=BLOCK, num_warps=8)
    return out


def _ours(packed, quant_state, dtype):
    s2 = getattr(quant_state, "state2", None)
    return (packed.is_cuda and quant_state.quant_type == "nf4" and quant_state.nested
            and s2 is not None and len(quant_state.shape) == 2
            and dtype in (torch.float16, torch.float32)
            and quant_state.absmax.dtype == torch.uint8)


def patch_bnb_dequant():
    import bitsandbytes.functional as bnbf

    try:
        from bitsandbytes.backends.cuda import ops as cuda_ops
    except Exception:  # no CUDA backend here: nothing of ours would run anyway
        cuda_ops = None

    original_dequant = bnbf.dequantize_4bit
    original_fallback = getattr(cuda_ops, "_dequant_linear_fallback", None)

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        # The [1, n] transposed form and out= go to bitsandbytes (its BC shim handles them).
        if quant_state is None or out is not None or A.shape[0] == 1 \
                or not _ours(A, quant_state, quant_state.dtype):
            return original_dequant(A, quant_state, absmax, out, blocksize, quant_type)
        s2 = quant_state.state2
        return nf4_dequant(A, quant_state.absmax, s2.absmax, s2.code, quant_state.offset,
                           quant_state.blocksize, s2.blocksize, quant_state.shape,
                           quant_state.dtype)

    def dequant_linear_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                                absmax_8bit=None, absmax_code=None, absmax_offset=None):
        if (absmax_8bit is None or quant_type != "nf4" or not A.is_cuda
                or A.dtype not in (torch.float16, torch.float32)):
            return original_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias,
                                     absmax_8bit, absmax_code, absmax_offset)
        # Called with absmax = state2.absmax (fp32, per 256 scales), absmax_8bit = the 8-bit
        # scales, absmax_code = state2.code, as MatMul4Bit passes them.
        weight = nf4_dequant(B, absmax_8bit, absmax, absmax_code, absmax_offset, blocksize,
                             256, tuple(shapeB), A.dtype)
        return torch.nn.functional.linear(A, weight, bias)

    bnbf.dequantize_4bit = dequantize_4bit
    if original_fallback is not None:
        cuda_ops._dequant_linear_fallback = dequant_linear_fallback


class _Prepared:
    """A 4-bit layer's dequantization arguments, flattened and typed once (they never change in
    LoRA training), so a step pays one kernel launch per dequantization and no Python around it."""

    __slots__ = ("state", "packed", "absmax8", "absmax2", "code8", "nf4", "offset", "blocksize",
                 "blocksize2", "shape", "n", "grid")

    def __init__(self, weight, state):
        s2 = state.state2
        self.state = state
        self.packed = weight.data.reshape(-1).view(torch.uint8)
        self.absmax8 = state.absmax.reshape(-1)
        self.absmax2 = s2.absmax.reshape(-1).float()
        self.code8 = s2.code.float()
        self.nf4 = _nf4_table(weight.device)
        offset = state.offset if torch.is_tensor(state.offset) else torch.tensor(float(state.offset))
        self.offset = offset.to(device=weight.device, dtype=torch.float32).reshape(1)
        self.blocksize, self.blocksize2 = state.blocksize, s2.blocksize
        self.shape = tuple(state.shape)
        self.n = self.shape[0] * self.shape[1]
        self.grid = (triton.cdiv(self.n, BLOCK),)

    def dequant(self, dtype):
        out = torch.empty(self.shape, device=self.packed.device, dtype=dtype)
        nf4_dequant_kernel[self.grid](self.packed, self.absmax8, self.absmax2, self.code8, self.nf4,
                                      self.offset, out, self.n, self.blocksize, self.blocksize2,
                                      BLOCK=BLOCK, num_warps=8)
        return out


class NF4Linear(torch.autograd.Function):
    """x @ dequant(W).T, the backward x-gradient with W dequantized again: bitsandbytes'
    dequantize + `F.linear` path for large batches, as one autograd node."""

    @staticmethod
    def forward(ctx, x, prepared):
        ctx.prepared = prepared
        return torch.nn.functional.linear(x, prepared.dequant(x.dtype))

    @staticmethod
    def backward(ctx, grad):
        return torch.matmul(grad, ctx.prepared.dequant(grad.dtype)), None


def _prepared(module):
    weight = module.weight
    state = getattr(weight, "quant_state", None)
    prepared = getattr(module, "_nf4_prepared", None)
    if prepared is not None and prepared.state is state:
        return prepared
    if (state is None or state.quant_type != "nf4" or not state.nested or not weight.is_cuda
            or getattr(state, "state2", None) is None or len(state.shape) != 2
            or state.absmax.dtype != torch.uint8 or state.dtype != torch.float16):
        return None
    prepared = _Prepared(weight, state)
    module._nf4_prepared = prepared
    return prepared


def patch_linear4bit_fast_path():
    """`bnb.nn.Linear4bit.forward` straight to `NF4Linear` exactly when bitsandbytes would
    dequantize and call `F.linear` (its own dispatch decides: `_gemm_4bit_use_custom_fn`), with an
    fp16 input already in the compute dtype, no bias and grad mode on. Otherwise the original."""
    import bitsandbytes as bnb
    from bitsandbytes.backends.cuda import ops as cuda_ops

    cls = bnb.nn.Linear4bit
    original = cls.forward
    use_custom = getattr(cuda_ops, "_gemm_4bit_use_custom_fn", None)
    max_m = getattr(cuda_ops, "_gemm_4bit_custom_max_m", None)

    def fallback_path(x, prepared):
        k = x.shape[-1]
        m = x.numel() // k
        if m > max_m or k % prepared.blocksize != 0:
            return True
        return not use_custom(x.device.index, x.dtype, m, prepared.shape[0], k)

    def forward(self, x):
        if (use_custom is not None and max_m is not None and torch.is_grad_enabled() and x.is_cuda
                and x.dtype == torch.float16 and self.bias is None
                and getattr(self, "compute_dtype", None) == torch.float16):
            prepared = _prepared(self)
            if prepared is not None and x.shape[-1] == prepared.shape[1] and fallback_path(x, prepared):
                return NF4Linear.apply(x, prepared)
        return original(self, x)

    cls.forward = forward
