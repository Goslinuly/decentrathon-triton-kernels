"""Hahathon Devs: our Triton kernels for Soup's training run.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.

Ours:
- `fused_head.py`, `head_patch.py`: the output head, the loss, TRL's entropy and token accuracy
  and Soup's per-row rewind losses without the [tokens, vocab] logits;
- `rmsnorm.py`: Qwen2RMSNorm forward and backward;
- `nf4.py`: NF4 dequantization of the 4-bit weights with the double-quantized scales;
- `mlp_fp16.py`: the MLP's input cast to fp16 once (no fp32 round trip per projection);
- `unpad.py`: pad tokens skipped in every layer's MLP, the fallback behind `packed_layers.py`
  (a supporting change in torch ops, allowed explicitly; idle while the packed stack runs);
- `tied_fp16.py`: the frozen tied embedding / lm_head weight stored in fp16, losslessly
  (933 MB less), the embedding still returning fp32;
- `packed_layers.py`: the whole decoder stack on the rows with attention, one varlen attention
  call per layer; only the LoRA dropout mask is drawn over the padded grid (same RNG draw).
- `lora_dropout.py`: the q/v LoRA dropout on the packed rows with the base's masks, kept as bits;
- `varlen_attention.py`: one causal varlen attention call per layer over the packed sequences;
- `deferred.py`: per-step metric and rewind reads moved to the log (no syncs before the backward);
- `fast_load.py`: checkpoint dtype conversion on the GPU during model load;
- `swiglu8.py`: SwiGLU that saves gate/up as blockwise int8 for the backward;
- `warmup.py`: the step's CUDA op families launched once in the background during model load.
`swiglu.py` is the organizers' example; no longer patched in (kept for its test).
"""

from .deferred import patch_deferred
from .fast_load import patch_fast_load
from .head_patch import patch_head
from .mlp_fp16 import patch_mlp_fp16
from .nf4 import patch_bnb_dequant, patch_linear4bit_fast_path
from .packed_layers import patch_packed_layers
from .precompile import start_precompile
from .rmsnorm import patch_qwen2_rmsnorm
from .swiglu8 import patch_qwen2_mlp_int8
from .timeline import install_timeline
from .tied_fp16 import patch_soup_setup
from .unpad import patch_unpad
from .warmup import start_gpu_warmup


def apply():
    install_timeline()  # analysis: phase timestamps in train.log
    start_precompile()  # our kernels compile in the background while Soup loads the model
    start_gpu_warmup()  # first launches of the step's CUDA ops, hidden behind the model load
    patch_fast_load()   # checkpoint bf16 -> fp16 on the GPU (bit-identical), not on the CPU
    patch_qwen2_mlp_int8()  # ours, in place of the example's SwiGLU patch
    patch_head()
    patch_deferred()   # metric/rewind syncs moved from before the backward to the step's log
    patch_qwen2_rmsnorm()
    patch_bnb_dequant()
    patch_linear4bit_fast_path()   # bnb's dequantize + F.linear path as one autograd node
    patch_mlp_fp16()   # after the example's MLP patch, inside the pad skipping
    patch_soup_setup()
    patch_packed_layers()  # Qwen2Model.forward: packed layers; falls back to the original
    patch_unpad()  # last: its wrappers must sit outside the patches above
    from .timeline import mark
    mark("apply end")
