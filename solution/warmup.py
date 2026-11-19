"""Background GPU warm-up of the step's CUDA op families while Soup loads the model.

precompile.py hides the Triton compiles, but the first training steps still pay one-time CUDA
costs that have nothing to do with our kernels: creating the context, lazily loading the cubin
of every CUDA op the step uses, the cuBLAS handle and workspace, and the first launches of the
attention, foreach-optimizer and grad-scaler kernels (step 0 runs ~2.9 s against the steady
~0.4 s). None of it needs the model: one daemon thread exercises each op family once on tiny
zero tensors, on its own CUDA stream, while the main thread loads and quantizes the model,
then frees everything it allocated.

Correctness guards:
- No RNG op runs here (no dropout), and the CUDA RNG state is saved and restored around the
  whole warm-up, so the philox offset the training step sees is untouched.
- Values are zeros and ones on throwaway tensors; nothing touches the model, the data, the
  allocator settings or any other global state.
- Every op group is wrapped: a failure only means that group warms up at step 0 as before.

Idea from the PyTorch tuning guide's note that first CUDA calls are expensive; our own code.
"""

import threading

_started = False


def _groups(torch):
    import torch.nn.functional as F

    from .varlen_attention import VarlenAttention

    dev = torch.device("cuda")
    f16, f32 = torch.float16, torch.float32

    def gemm_fwd_bwd():
        # the projections, LoRA matmuls and head chunks: fp16 mm forward and backward
        a = torch.ones(256, 1536, device=dev, dtype=f16, requires_grad=True)
        w = torch.ones(1536, 1536, device=dev, dtype=f16, requires_grad=True)
        (a @ w).float().sum().backward()

    def packing():
        # unpad/packed_layers: nonzero, index_select, scatter back into the grid, cu_seqlens
        mask = torch.ones(4, 64, device=dev, dtype=torch.int64)
        idx = mask.flatten().nonzero(as_tuple=False).squeeze(1)
        rows = torch.ones(256, 1536, device=dev, dtype=f32).index_select(0, idx)
        torch.zeros(256, 1536, device=dev, dtype=f32).index_copy_(0, idx, rows)
        mask.sum(-1).cumsum(0)

    def embedding_and_residual():
        table = torch.ones(1024, 1536, device=dev, dtype=f16)
        emb = F.embedding(torch.zeros(64, device=dev, dtype=torch.int64), table).to(f32)
        (emb + emb).to(f16)

    def attention():
        # the step's one varlen attention call per layer, forward and backward
        q = torch.ones(1, 64, 12, 128, device=dev, dtype=f16, requires_grad=True)
        k = torch.ones(1, 64, 12, 128, device=dev, dtype=f16)
        v = torch.ones(1, 64, 12, 128, device=dev, dtype=f16)
        cu = torch.tensor([0, 32, 64], device=dev, dtype=torch.int32)
        out = VarlenAttention.apply(q, k, v, cu, 32, 0.0883883476483184)
        out.float().sum().backward()

    def optimizer():
        # adamw_torch runs on the foreach kernels; grad clipping on _foreach_norm
        p = torch.zeros(64, device=dev, dtype=f32, requires_grad=True)
        p.grad = torch.zeros_like(p)
        torch.optim.AdamW([p], lr=1e-3, foreach=True).step()
        torch._foreach_norm([p.grad])

    def grad_scaler():
        g = [torch.zeros(8, device=dev, dtype=f32)]
        found = torch.zeros(1, device=dev, dtype=f32)
        torch._amp_foreach_non_finite_check_and_unscale_(g, found, torch.ones(1, device=dev))
        scale = torch.full((1,), 65536.0, device=dev, dtype=f32)
        growth = torch.zeros(1, device=dev, dtype=torch.int32)
        torch._amp_update_scale_(scale, growth, found, 2.0, 0.5, 2000)

    return [gemm_fwd_bwd, packing, embedding_and_residual, attention, optimizer, grad_scaler]


def _run():
    import torch

    if not torch.cuda.is_available():
        return
    rng_state = torch.cuda.get_rng_state()
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            for group in _groups(torch):
                try:
                    group()
                except Exception as exc:
                    print(f"[solution] GPU warm-up {group.__name__} skipped: {exc}", flush=True)
            torch.cuda.synchronize()
    finally:
        torch.cuda.set_rng_state(rng_state)
        torch.cuda.empty_cache()


def _safe_run():
    try:
        _run()
    except Exception as exc:
        print(f"[solution] GPU warm-up skipped: {exc}", flush=True)


def start_gpu_warmup():
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_safe_run, name="gpu-warmup", daemon=True).start()
