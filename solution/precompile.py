"""Compile our Triton kernels in a background thread while Soup loads the model.

Every measured run starts with an empty Triton cache, so each of our kernels is compiled on its
first launch, inside the first training steps (~5 s of the loop in our runs). Triton 3.6's
`JITFunction.warmup` compiles a kernel for a given signature without launching it; here one
daemon thread, started by `apply()`, warms every kernel with the exact argument types,
specializations and launch options of the training step, while the main thread imports Soup
and loads and quantizes the model. If a warm-up does not match a real launch, that launch
simply compiles as before; nothing here runs on the GPU or changes any result.

The same thread then builds each kernel's C launcher and loads its cubin, which Triton otherwise
does lazily on the first launch inside step 0.

Pattern from vLLM's `jit_warmup_triton_helper` and Triton's `AsyncCompileMode`; our own code.
"""

import threading

_started = False


def _jobs():
    import torch

    from .fused_head import BLOCK_V, head_ce_rows_kernel
    from .lora_dropout import BLOCK as DROP_BLOCK, lora_dropout_bwd_kernel, lora_dropout_fwd_kernel
    from .nf4 import BLOCK as NF4_BLOCK, nf4_dequant_kernel
    from .rmsnorm import rmsnorm_bwd_kernel, rmsnorm_fwd_kernel
    from .swiglu8 import (QBLOCK, QBLOCKS_PER_PROG, ROWS_PER_PROG, swiglu8_bwd_kernel,
                          swiglu8_fwd_kernel)

    f16, f32, i8, u8, i32, i64 = (torch.float16, torch.float32, torch.int8, torch.uint8,
                                  torch.int32, torch.int64)
    hidden, vocab, inter = 1536, 151936, 8960
    blocks = (inter + QBLOCK - 1) // QBLOCK
    norm_block = 2048
    swiglu_meta = dict(ROWS=ROWS_PER_PROG, QBLOCK=QBLOCK, NB=QBLOCKS_PER_PROG, num_warps=4)
    return [
        (head_ce_rows_kernel, (f16, i64, f32, f32, i32, f32, 512, vocab, vocab, 1.0),
         dict(BLOCK_V=BLOCK_V, num_warps=8)),
        (rmsnorm_fwd_kernel, (f32, f32, f32, f16, f32, hidden, 1e-6, hidden, hidden),
         dict(ROUND_TO_INPUT=False, SAVE_HALF=True, BLOCK=norm_block, num_warps=8)),
        (rmsnorm_bwd_kernel, (f32, f16, f32, f32, f32, hidden, hidden, hidden, hidden),
         dict(BLOCK=norm_block, num_warps=8)),
        (nf4_dequant_kernel, (u8, u8, f32, f32, f32, f32, f16, hidden * hidden, 64, 256),
         dict(BLOCK=NF4_BLOCK, num_warps=8)),
        (swiglu8_fwd_kernel, (f16, f16, f16, i8, i8, f32, f32, 1024, inter, blocks), swiglu_meta),
        (lora_dropout_fwd_kernel, (f32, u8, i64, f16, u8, 1024, hidden, 1.0 / 0.95),
         dict(BLOCK=DROP_BLOCK, num_warps=4)),
        (lora_dropout_bwd_kernel, (f16, u8, f32, 1024, hidden, 1.0 / 0.95),
         dict(BLOCK=DROP_BLOCK, num_warps=4)),
        (lora_dropout_bwd_kernel, (f16, u8, f16, 1024, hidden, 1.0 / 0.95),
         dict(BLOCK=DROP_BLOCK, num_warps=4)),
        (swiglu8_bwd_kernel, (f16, i8, i8, f32, f32, f16, f16, 1024, inter, blocks), swiglu_meta),
    ]


def _warm():
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.set_device(torch.cuda.current_device())   # a current context for this thread
        for kernel, args, meta in _jobs():
            try:
                compiled = kernel.warmup(*args, grid=(1,), **meta)
            except Exception as exc:  # a mismatch only means that launch compiles later
                print(f"[solution] warm-up of {kernel.fn.__name__} skipped: {exc}", flush=True)
                continue
            try:
                # `warmup` stops at the compiled kernel; the C launcher is built and the cubin
                # loaded lazily on the first launch (`CompiledKernel._init_handles`), on the
                # step's critical path. Touching `run` does it here, on the same thread.
                getattr(compiled, "run", None)
            except Exception as exc:
                print(f"[solution] launcher of {kernel.fn.__name__} left to the first launch: "
                      f"{exc}", flush=True)
    except Exception as exc:
        print(f"[solution] Triton warm-up skipped: {exc}", flush=True)


def start_precompile():
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_warm, name="triton-warmup", daemon=True).start()
