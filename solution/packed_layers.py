"""The decoder layers on the rows that have attention: no pad tokens anywhere but in dropout.

`unpad.py` skips the pad rows in the MLP. Everything else in a layer still runs over the padded
[batch, seq] grid: both norms, q/k/v/o, the rotary embedding, the residual adds and the attention
itself, whose cost grows with the padded length squared. Here `Qwen2Model.forward` packs the
rows with attention once, right after the embedding, runs every layer on the packed [rows, dim]
tensor and unpacks once before the head. Attention is one causal varlen call per layer over
all the packed sequences (`varlen_attention.py`, PyTorch's memory-efficient kernel with
`cu_seqlens`): inside each sequence exactly the keys the base's causal + padding mask lets
through, so the values match the base to rounding. If that kernel is not available, the same
attention runs per sequence on its own contiguous slice with `is_causal=True`.

The one thing that still looks at the padded grid is the LoRA dropout of q_proj and v_proj:
dropout draws its mask from the RNG by element count, so the mask is drawn over the padded
[batch * seq, dim] shape (`lora_dropout.py`: the same `native_dropout` call as the base, hence
the same mask on every real row and the same RNG stream for the rest of the step) and applied
on the packed rows only by our kernel. The LoRA matmuls (rank 16) and the 4-bit base projection
run packed. This is the vanilla LoRA path of peft 0.20 (`Linear4bit.forward`:
`result + lora_B(lora_A(dropout(x))) * scaling`) written out for packed rows; any other
configuration (merged, disabled, variants, mixed batches) is not packed at all.

Every assumption is checked up front (`packing_plan`): training forward with labels, 2D
attention mask with right padding and at least one pad token, no cache, default positions,
full-attention SDPA layers, vanilla LoRA, autocast on. Anything else takes the original forward
untouched: evaluation, generation, Soup's tests and a batch without padding.

Removing pad tokens inside a batch is allowed explicitly; rows of different batches are never
mixed. Ideas: padding-free training (Hugging Face `DataCollatorWithFlattening`, Unsloth); the
code is our own.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .lora_dropout import packed_dropout, packed_lora
from .varlen_attention import varlen_attention, varlen_ok

MODEL_KWARGS_ALLOWED = frozenset({"num_items_in_batch", "output_attentions",
                                  "output_hidden_states", "return_dict", "cache_position"})


@dataclass(frozen=True)
class Plan:
    rows: torch.Tensor        # indices of the rows with attention in the [batch * seq] grid
    segments: tuple           # (start, length) of each sequence inside the packed rows
    positions: torch.Tensor   # [1, rows] position of every packed row inside its sequence
    batch: int
    seq: int
    cu_seqlens: torch.Tensor = None   # [segments + 1] int32 offsets of the sequences (varlen attention)
    max_len: int = 0


def packing_plan(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
                 use_cache, kwargs):
    """The packing of this forward, or None when the original forward must run."""
    if not (isinstance(model, torch.nn.Module) and model.training and torch.is_autocast_enabled()):
        return None
    if input_ids is None or inputs_embeds is not None or past_key_values is not None or use_cache:
        return None
    if position_ids is not None or kwargs.get("cache_position") is not None:
        return None
    if any(key not in MODEL_KWARGS_ALLOWED for key in kwargs):
        return None
    if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
        return None
    mask = attention_mask
    if not torch.is_tensor(mask) or mask.dim() != 2 or not mask.is_cuda or mask.shape != input_ids.shape:
        return None
    if getattr(model, "gradient_checkpointing", False) or not _layers_packable(model):
        return None
    batch, seq = mask.shape
    # One device sync for the whole plan: the mask is 2,048 bools, the rest is CPU work.
    keep = mask.bool().cpu()
    lengths = keep.sum(dim=1)
    arange = torch.arange(seq)
    if not bool((keep == (arange[None, :] < lengths[:, None])).all()):
        return None  # not right-padded: the original mask semantics are needed
    lengths_list = lengths.tolist()
    total = sum(lengths_list)
    if total == batch * seq or total == 0:
        return None
    segments, start = [], 0
    for length in lengths_list:
        if length:
            segments.append((start, length))
            start += length
    rows = keep.reshape(-1).nonzero().squeeze(1)
    positions = torch.cat([arange[:length] for length in lengths_list if length]).unsqueeze(0)
    device = mask.device
    cu = torch.tensor([0] + [start + length for start, length in segments], dtype=torch.int32)
    return Plan(rows=rows.to(device, non_blocking=True), segments=tuple(segments),
                positions=positions.to(device, non_blocking=True), batch=batch, seq=seq,
                cu_seqlens=cu.to(device, non_blocking=True),
                max_len=max(length for _, length in segments))


def _vanilla_lora(module):
    """A peft LoRA layer on the plain path, or a module without adapters (None = not packable)."""
    if not hasattr(module, "lora_A"):
        return "plain"
    needed = ("base_layer", "lora_A", "lora_B", "lora_dropout", "scaling", "active_adapters")
    if not all(hasattr(module, name) for name in needed):
        return None
    if getattr(module, "merged", False) or getattr(module, "disable_adapters", False):
        return None
    variants = getattr(module, "lora_variant", {}) or {}
    for adapter in module.active_adapters:
        if adapter in module.lora_A and adapter in variants:
            return None
    return "lora"


def _layers_packable(model):
    config = getattr(model, "config", None)
    layers = getattr(model, "layers", None)
    if config is None or layers is None or not hasattr(model, "rotary_emb") or not hasattr(model, "norm"):
        return False
    if getattr(config, "_attn_implementation", None) != "sdpa" or getattr(model, "has_sliding_layers", False):
        return False
    n_layers = config.num_hidden_layers
    layer_types = getattr(config, "layer_types", None) or ["full_attention"] * n_layers
    if any(kind != "full_attention" for kind in layer_types[:n_layers]):
        return False
    for layer in layers[:n_layers]:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "sliding_window", None) is not None:
            return False
        if not getattr(attn, "is_causal", True) or float(getattr(attn, "attention_dropout", 0.0)) != 0.0:
            return False
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if _vanilla_lora(getattr(attn, name, None)) is None:
                return False
        for name in ("input_layernorm", "post_attention_layernorm", "mlp"):
            if getattr(layer, name, None) is None:
                return False
    return True


def _padded(packed, plan):
    """The packed rows scattered back into the [batch * seq, dim] grid, zeros on pad rows:
    what the LoRA dropout must see to draw the same mask as the base."""
    return torch.zeros(plan.batch * plan.seq, packed.shape[-1], device=packed.device,
                       dtype=packed.dtype).index_copy(0, plan.rows, packed)


def _fp16_compute(module):
    base = getattr(module, "base_layer", module)
    return getattr(base, "compute_dtype", None) == torch.float16


def _projection(module, packed, plan, padded=None, packed16=None):
    """`module(x)` on the packed rows. For a LoRA layer: the base projection packed, the
    dropout on the padded grid (same RNG draw as the base), the rank-16 matmuls packed.
    `padded` is the grid of `packed` when the caller already has it (shared by q and v).
    `packed16` is `packed` cast once to the 4-bit layers' fp16 compute dtype (shared by q, k, v):
    the base projection gets it instead of casting its own copy. A LoRA layer's base output is
    upcast to `packed`'s dtype, as bitsandbytes' `.to(inp_dtype)` does, so the adapter sum is the
    base's; a plain layer's fp16 output only meets fp32 cos/sin (exact upcast) or SDPA's fp16 cast."""
    base_in = packed if packed16 is None else packed16
    if not hasattr(module, "lora_A"):
        return module(base_in)
    result = module.base_layer(base_in)
    result = result.to(packed.dtype) if result.dtype != packed.dtype else result.clone()
    for adapter in module.active_adapters:
        if adapter not in module.lora_A:
            continue
        lora_a, lora_b = module.lora_A[adapter], module.lora_B[adapter]
        dropout, scaling = module.lora_dropout[adapter], module.scaling[adapter]
        if packed16 is not None:   # keeps packed16 (shared) instead of a dropped copy
            branch = packed_lora(dropout, packed, packed16, plan, lora_a, lora_b, scaling)
            if branch is not None:
                result = result + branch
                continue
        dropped = packed_dropout(dropout, packed, plan)   # the base's mask, packed rows only
        if dropped is None:
            if padded is None:
                padded = _padded(packed, plan)
            dropped = dropout(padded).index_select(0, plan.rows)
        result = result + lora_b(lora_a(dropped)) * scaling
    return result


def _attention(attn, hidden, cos, sin, plan, rotary, repeat_kv):
    rows = hidden.shape[0]
    head_dim = attn.head_dim
    grid = None   # built only if a projection's dropout cannot take `packed_dropout`
    projections = (attn.q_proj, attn.k_proj, attn.v_proj)
    hidden16 = (hidden.to(torch.float16) if hidden.dtype == torch.float32
                and all(_fp16_compute(m) for m in projections) else None)
    query = _projection(attn.q_proj, hidden, plan, grid, hidden16).reshape(1, rows, -1, head_dim)
    key = _projection(attn.k_proj, hidden, plan, grid, hidden16).reshape(1, rows, -1, head_dim)
    value = _projection(attn.v_proj, hidden, plan, grid, hidden16).reshape(1, rows, -1, head_dim)
    groups = getattr(attn, "num_key_value_groups", 1)
    if varlen_ok(query, plan):
        # [1, rows, heads, dim] all the way: rotary broadcasts over dim 2, one varlen call.
        query, key = rotary(query, key, cos, sin, unsqueeze_dim=2)
        out = varlen_attention(query, key, value, plan.cu_seqlens, plan.max_len, attn.scaling,
                               groups)
        return _projection(attn.o_proj, out.reshape(rows, -1), plan)
    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
    query, key = rotary(query, key, cos, sin)
    if groups > 1:
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
    pieces = []
    for start, length in plan.segments:
        stop = start + length
        pieces.append(F.scaled_dot_product_attention(
            query[:, :, start:stop], key[:, :, start:stop], value[:, :, start:stop],
            attn_mask=None, dropout_p=0.0, scale=attn.scaling, is_causal=True))
    out = torch.cat(pieces, dim=2).transpose(1, 2).reshape(rows, -1).contiguous()
    return _projection(attn.o_proj, out, plan)


def _decoder_layer(layer, hidden, cos, sin, plan, rotary, repeat_kv):
    residual = hidden
    hidden = layer.input_layernorm(hidden)
    hidden = _attention(layer.self_attn, hidden, cos, sin, plan, rotary, repeat_kv)
    hidden = residual + hidden
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    hidden = layer.mlp(hidden)
    return residual + hidden


_original_forward = None   # the pristine Qwen2Model.forward, kept for the fallback and the tests


def patch_packed_layers():
    global _original_forward
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Model
    if _original_forward is not None:   # already installed: patching twice would nest wrappers
        return
    original = _original_forward = cls.forward
    rotary = modeling_qwen2.apply_rotary_pos_emb
    repeat_kv = modeling_qwen2.repeat_kv

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        plan = packing_plan(self, input_ids, attention_mask, position_ids, past_key_values,
                            inputs_embeds, use_cache, kwargs)
        if plan is None:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        embeds = self.embed_tokens(input_ids)
        dim = embeds.shape[-1]
        hidden = embeds.reshape(plan.batch * plan.seq, dim).index_select(0, plan.rows)
        cos, sin = self.rotary_emb(hidden.unsqueeze(0), plan.positions)
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden = _decoder_layer(layer, hidden, cos, sin, plan, rotary, repeat_kv)
        hidden = self.norm(hidden)
        full = torch.zeros(plan.batch * plan.seq, dim, device=hidden.device,
                           dtype=hidden.dtype).index_copy(0, plan.rows, hidden)
        return BaseModelOutputWithPast(last_hidden_state=full.view(plan.batch, plan.seq, dim),
                                       past_key_values=None)

    cls.forward = forward
