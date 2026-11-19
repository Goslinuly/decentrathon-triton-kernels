"""Pad tokens skipped in the MLP of every decoder layer.

By our count of the public rows (median 140 tokens, batches padded to the longest row, often
512), about 40% of the positions in a training batch are padding. The MLP's three projections
(1536 -> 8960 twice and 8960 -> 1536) are the FLOP-heaviest per-token work of a step, and a pad
token's MLP output is never read: attention masks pad keys out, the head and the metrics skip
positions without attention, and a pad position's hidden state only ever feeds other pad
positions. The task statement explicitly allows removing pad tokens inside a batch.

So, on a training forward with labels, the rows with attention are gathered once per batch
(`index_select`), each layer's MLP runs on the packed rows only, and the result is scattered
back (`index_copy`) with zeros for the pad rows (the base writes garbage there; nobody reads
either). Attention, the LoRA projections and their dropout see the exact padded tensors they
always did, so the dropout RNG stream and the loss curve match the base. The MLP has no
dropout and no adapters in this task (LoRA is on q_proj and v_proj), so packing it changes
nothing but the row count.

This is a supporting change around our kernels, in plain differentiable torch ops: the packed
rows then flow through the fused SwiGLU and our NF4 dequantization. Idea sources: padding-free
training as in Hugging Face's `DataCollatorWithFlattening` and Unsloth's padding-free path
(https://huggingface.co/blog/packing-with-FA2, https://github.com/unslothai/unsloth) - the
code is our own, and we do not mix rows across batches.
"""

import threading

import torch

_state = threading.local()


def _rows_for(x):
    """Kept-row indices for a padded [batch, seq, dim] MLP input, derived from the step's mask
    once and only when an MLP actually sees the padded grid (the packed layers hand it 2D
    tensors, so in that case nothing here touches the device); None = nothing to skip."""
    mask = getattr(_state, "mask", None)
    if mask is None or mask.numel() != x.shape[0] * x.shape[1]:
        return None
    rows = getattr(_state, "rows", None)
    if rows is None:
        keep = mask.reshape(-1).bool()
        rows = keep.nonzero().squeeze(1) if not bool(keep.all()) else False
        _state.rows = rows
    return None if rows is False else rows


def patch_unpad():
    """Two class patches, applied last so they wrap whatever is installed:

    - `Qwen2ForCausalLM.forward`: on a training step with labels and a 2D attention mask,
      leave the mask for the layers below; always cleared.
    - `Qwen2MLP.forward`: given a padded [batch, seq, dim] input (the packed layers of
      `packed_layers.py` did not run), derive the kept rows once, run the (already patched) MLP
      on them and scatter the result back. Any other call - eval, generation, Soup's tests, a
      batch with no padding, a 2D input - takes the original path untouched.

    This is the fallback behind `packed_layers.py`, which packs the whole stack; it costs nothing
    while that path is active.
    """
    from transformers.models.qwen2 import modeling_qwen2

    lm_cls = modeling_qwen2.Qwen2ForCausalLM
    inner_lm_forward = lm_cls.forward

    def lm_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        mask = attention_mask
        if (isinstance(self, torch.nn.Module) and getattr(self, "training", False)
                and labels is not None and torch.is_tensor(mask) and mask.dim() == 2
                and mask.is_cuda and kwargs.get("logits_to_keep", 0) == 0):
            _state.mask, _state.rows = mask, None   # rows only if an MLP asks for them
        try:
            return inner_lm_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                    labels=labels, **kwargs)
        finally:
            _state.mask = _state.rows = None

    lm_cls.forward = lm_forward

    mlp_cls = modeling_qwen2.Qwen2MLP
    inner_mlp_forward = mlp_cls.forward

    def mlp_forward(self, x):
        if not torch.is_tensor(x) or x.dim() != 3 or not x.is_cuda:
            return inner_mlp_forward(self, x)
        rows = _rows_for(x)
        if rows is None:
            return inner_mlp_forward(self, x)
        batch, seq, dim = x.shape
        flat = x.reshape(batch * seq, dim)
        packed = inner_mlp_forward(self, flat.index_select(0, rows))
        # The residual stream can be fp32 while autocast makes the MLP's output fp16: scatter
        # in the output's dtype, and the residual add outside promotes exactly as in the base.
        full = torch.zeros_like(flat, dtype=packed.dtype).index_copy(0, rows, packed)
        return full.view(batch, seq, dim)

    mlp_cls.forward = mlp_forward
