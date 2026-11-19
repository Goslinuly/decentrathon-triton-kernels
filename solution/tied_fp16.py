"""The tied embedding / lm_head weight kept in fp16: lossless, 933 MB less.

The checkpoint stores `embed_tokens.weight` (tied to `lm_head.weight`) in fp16.
`prepare_model_for_kbit_training` upcasts every non-quantized parameter to fp32, so the base
run holds a 151,936 x 1536 fp32 copy (933 MB) of values that are exactly representable in
fp16. The weight is frozen in LoRA training, the embedding lookup is a gather (no arithmetic),
and the head's matmul under fp16 autocast casts the weight to fp16 anyway. So the weight is
stored in fp16 and the embedding output is upcast to fp32 - bit for bit what the base produces
- and the fused head uses the weight directly instead of a second fp16 copy (another 466 MB).

The cast happens only when it is lossless (checked on the tensor itself), only for a frozen
fp32 embedding, and only after Soup has built the final PEFT model. Weight repacking of a
frozen weight; nothing trainable changes and no output changes.
"""

import torch

LOSSLESS_DTYPE = torch.float16


CHUNK_BYTES = 64 << 20   # of the fp32 source per round-trip check


def _lossless_half(weight):
    """fp16 copy of `weight` when every value round-trips exactly, else None.

    Converted and checked in chunks of about 64 MB: a whole-tensor check would hold the fp32
    weight, its fp16 copy and a second fp32 copy at once (2.3 GB for the 152k x 1536 embedding),
    and that moment, not the training step, would set the run's peak memory."""
    source = weight.detach()
    rows = source.shape[0]
    step = max(1, CHUNK_BYTES // max(1, source[0].numel() * source.element_size()))
    half = torch.empty(source.shape, device=source.device, dtype=LOSSLESS_DTYPE)
    for start in range(0, rows, step):
        src, dst = source[start:start + step], half[start:start + step]
        dst.copy_(src)                                   # round to nearest even, as .to() does
        if not torch.equal(dst.to(source.dtype), src):
            del half
            return None
    return half


def shrink_tied_embedding(model):
    """Store the frozen fp32 input embedding (and a tied or frozen lm_head) in fp16; the
    embedding keeps returning fp32. Returns the number of parameters converted."""
    if not isinstance(model, torch.nn.Module):
        return 0
    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    embed = get_in() if callable(get_in) else None
    head = get_out() if callable(get_out) else None
    if not isinstance(embed, torch.nn.Embedding):
        return 0
    weight = embed.weight
    if weight.requires_grad or weight.dtype != torch.float32 or not weight.is_cuda:
        return 0
    half = _lossless_half(weight)
    if half is None:
        return 0

    converted = 1
    head_weight = getattr(head, "weight", None) if isinstance(head, torch.nn.Module) else None
    if head_weight is weight:
        weight.data = half
    else:
        weight.data = half
        if (isinstance(head_weight, torch.Tensor) and not head_weight.requires_grad
                and head_weight.dtype == torch.float32 and head_weight.is_cuda):
            head_half = _lossless_half(head_weight)
            if head_half is not None:
                head_weight.data = head_half
                converted += 1

    out_dtype = torch.float32
    embedding_forward = type(embed).forward

    def forward(self, input):
        return embedding_forward(self, input).to(out_dtype)

    embed.forward = forward.__get__(embed, type(embed))
    return converted


def patch_soup_setup():
    """After `SFTTrainerWrapper._setup_transformers` has built the final PEFT model, shrink
    its tied embedding. Any other model (another backend, a test double) is left alone."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    original = cls._setup_transformers

    def _setup_transformers(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            shrink_tied_embedding(getattr(self, "model", None))
        except Exception as exc:  # noqa: BLE001 - a failed shrink must never fail the run
            print(f"[solution] tied embedding kept in fp32: {type(exc).__name__}: {exc}")
        return result

    cls._setup_transformers = _setup_transformers
