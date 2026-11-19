# Triton kernels for LoRA fine-tuning on a T4 — Soup Kernel Challenge (Decentrathon 6.0)

Custom Triton kernels that make `soup train` (Soup 0.77.0: Qwen2.5-1.5B-Instruct, 4-bit NF4
weights, LoRA r=16) **1.6–1.7x faster and 4.9x lighter on GPU memory** on a free Kaggle T4,
while training the same adapter (eval loss within 0.01 % of the baseline).

**Project page:** https://goslinuly.github.io/decentrathon-triton-kernels/

## Results (Kaggle T4, `harness/run_task.py`)

Final version (`v11-admitted`), task rows, seed 1, one run per side:

| | Baseline Soup 0.77.0 | This solution | Ratio |
|---|---:|---:|---:|
| Whole `soup train` command | 94.5 s | **55.6 s** | speedup **1.70x** |
| 64 training steps / median step | 61.4 s / 0.977 s | 33.0 s / 0.450 s | 1.86x / 2.17x |
| Peak GPU memory (nvidia-smi) | 14 345 MiB | **2 929 MiB** | **4.90x** less |
| Eval loss, held-out rows 40000–40255 | 2.04965 | 2.04966 | +0.000 % |
| Loss curve vs baseline: first step / mean | — | — | 0.03 % / 0.01 % (limit 1 % / 3 %) |
| Share of our kernels in training GPU time | — | 16.2 % | rule: ≥ 5 % |

All 10 rule checks of `harness/check_rules.py` pass; 35 kernel tests and Soup's 79 regression
tests pass with the patches applied.

**Jury-like check** (same code, train on rows 40000–40255, eval on 0–255, seed 7,
base/solution/solution/base, mean of two runs): baseline 88.2 s / 14 369 MiB, ours
**56.4 s / 3 145 MiB** → speedup **1.56x**, memory **4.57x**, eval loss +0.007 %.
The baseline's model-load time varies between Kaggle sessions (24–34 s), so single-run speedups of
the same code ranged 1.55–1.74x; the training loop (31–33 s) and the memory are stable.

## The problem

Profile of the baseline on this task: **44 %** of GPU time goes to element-wise ops and dtype casts,
**7 %** to softmax/cross-entropy over a 151 936-token vocabulary (fp32 logits are gigabytes),
**5 %** to NF4 dequantization; memory peaks at 14.3 of the T4's 15 GB.

Key finding: **Triton 3.6 does not use the T4's tensor cores** (sm_75, upstream
triton-lang/triton#5066). Our first fused head with `tl.dot` was 9x slower than the baseline
(359 ms vs 15 ms for the same matmul in cuBLAS). So every matmul stays in cuBLAS and Triton takes
over everything *around* the matmuls.

## Approach: 8 `@triton.jit` kernels

All in `solution/`, installed by a single `solution.apply()` call before Soup builds the model.

| Kernel | File | What it replaces / why |
|---|---|---|
| `head_ce_rows_kernel` | `fused_head.py` | Full `[tokens, vocab]` fp32 logits + CE + TRL entropy/accuracy + Soup's per-row "rewind" losses. Online softmax per row in one pass over 256-row logit chunks; the gradient is written over the chunk in place (biggest memory win). |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `rmsnorm.py` | `Qwen2RMSNorm` (~6 PyTorch kernels per call, 57 norms per step); the backward input is kept in fp16. |
| `nf4_dequant_kernel` | `nf4.py` | bitsandbytes' two-kernel NF4 double-quant dequantization, as one kernel in forward and backward of every 4-bit layer. |
| `swiglu8_fwd_kernel`, `swiglu8_bwd_kernel` | `swiglu8.py` | SwiGLU MLP; `gate`/`up` saved for backward as **int8 with one fp32 scale per 64 values** (activation compression). |
| `lora_dropout_fwd_kernel`, `lora_dropout_bwd_kernel` | `lora_dropout.py` | q/v LoRA dropout on packed rows using the baseline's RNG masks, stored as bits; the whole LoRA branch is one autograd node that keeps the layer's fp16 input once. |

Supporting (non-Triton) changes, allowed by the rules: the whole decoder stack runs only on
non-pad rows (`packed_layers.py`, `unpad.py`), one varlen attention call per layer with k/v kept
on the 2 kv-heads for backward (`varlen_attention.py`), lossless fp16 tied embedding
(`tied_fp16.py`, −933 MB), MLP input cast to fp16 once (`mlp_fp16.py`), metric/rewind syncs moved
to the end of the step (`deferred.py`), background Triton precompilation and GPU warm-up during
model load (`precompile.py`, `warmup.py`), bf16→fp16 checkpoint conversion on the GPU
(`fast_load.py`). The task itself is unchanged — same model, data, batches, LoRA config, optimizer
and Soup outputs (verified by `check_rules.py`).

Ideas credited (code is our own): Liger-Kernel fused linear cross-entropy; Cut Cross-Entropy
(Wijmans et al. 2024, arXiv:2411.09009); online softmax (Milakov & Gimelshein 2018,
arXiv:1805.02867); RMSNorm (Zhang & Sennrich 2019) and the Triton LayerNorm tutorial; 8-bit
blockwise quantization (Dettmers et al. 2021, arXiv:2110.02861); ActNN (Chen et al. 2021,
arXiv:2104.14129); bitsandbytes reference NF4 layout; padding-free training (HF
`DataCollatorWithFlattening`, Unsloth); vLLM's Triton warm-up pattern.

## Reproduce

**Kaggle (how the jury measured it), ~10 minutes:**

1. Create → New Notebook → File → Import Notebook → `templates/kaggle_notebook.ipynb`.
2. Settings: **Accelerator: GPU T4 x2**, **Internet: on**.
3. **Run All.** The notebook clones this repo, installs the pinned stack, prepares data, runs the
   kernel tests, trains baseline and solution, evaluates both adapters and runs the rule checks.
   Expected tail: `speedup ~1.7x  memory base/solution ~4.9x` and `ALL CHECKS PASSED`.

On Colab with a T4 runtime, run the commands from the local section below instead.

**Locally (Linux + CUDA GPU; only T4 numbers are comparable):**

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q                     # kernel tests vs PyTorch references (GPU)
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q  # Soup's 79 regression tests with our patches
python harness/run_task.py                       # baseline, then solution -> runs/result.json
python harness/check_rules.py                    # 10 rule checks -> runs/checks.json
```

**Without a GPU:** `python -m pytest` runs the static repository checks in `tests_cpu/`; GPU kernel
tests skip (or are not collected when `torch`/`triton` are absent), Soup tests run when `soup-cli`
is installed.

## Repository layout

```
solution/   our kernels and patches (+ test_*.py against PyTorch references)
harness/    organizers' runner, rule checks and scorer (unchanged)
tests/      Soup v0.77.0 regression tests (unchanged)
task/       fixed task config and data preparation (unchanged)
templates/  Kaggle notebook to reproduce the measurement
tools/      profiling / timeline analysis helpers
tests_cpu/  static checks that run without a GPU
docs/       detailed write-up (Russian) and the organizers' starter README
site/       GitHub Pages project page
```

## Limits and next steps

- Built for the challenge's conditions: Kaggle T4 (fp16, sm_75) and the pinned stack. Every patch
  checks its assumptions and falls back to Soup's original path otherwise.
- Matmuls intentionally stay in cuBLAS (see the tensor-core finding above).
- Triton compile time counts in the measurement; background precompilation hides most of it.
- ~22 s of the ~56 s run is model load/quantization, identical for both sides. Next: a fused LoRA
  branch (dropout + rank-16 matmuls in one pass) and parallel model loading.

## Hackathon context

Built in 36 hours at **Decentrathon 6.0** (Oct 30–31, 2026), Soup Kernel Challenge case, by team
**Hahathon Devs**. The final version passed all of the organizers' admission checks
(tag `v11-admitted`). Triton kernels, varlen attention and profiling: Nursultan Dauletkan
([github.com/Goslinuly](https://github.com/Goslinuly)). AI coding assistants were used, as allowed
by the rules.

## Кратко по-русски

Свои Triton-кернелы для `soup train` (Qwen2.5-1.5B, NF4, LoRA r=16) на Kaggle T4: **в 1,6–1,7 раза
быстрее и в 4,9 раза меньше пиковой памяти GPU** при том же качестве адаптера. Главная находка —
Triton 3.6 не использует тензорные ядра T4, поэтому матмулы оставлены cuBLAS, а в Triton перенесено
всё вокруг них: голова модели с онлайн-softmax без материализации логитов, RMSNorm, NF4-деквантизация,
SwiGLU с int8-активациями, LoRA-dropout с битовой маской. Подробное описание — `docs/DETAILS_RU.md`.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`), same as the organizers' starter kit
[soup-kernel-challenge](https://github.com/MakazhanAlpamys/soup-kernel-challenge), whose
`harness/`, `tests/`, `task/` and notebook template are included. The model
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0) and the
dataset [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT) are downloaded at run time and not shipped here.
