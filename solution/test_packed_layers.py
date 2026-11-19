"""The packed decoder layers against the padded original on a small Qwen2 with LoRA: same
hidden states on the kept rows with the same dropout masks (same seed), zeros on pad rows, same
LoRA gradients, a visibly different result under another seed (dropout really runs), and the
untouched original on left padding. Run on a GPU: `python -m pytest solution -q`."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quantize_linears(model):
    """The seven projections of every layer as bitsandbytes NF4 `Linear4bit`, as in the task
    (the fp16 compute dtype, double quantization); quantized when the model moves to the GPU."""
    import bitsandbytes as bnb

    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in PROJECTIONS and isinstance(child, torch.nn.Linear):
                quant = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=False,
                                          compute_dtype=torch.float16, quant_type="nf4",
                                          compress_statistics=True)
                quant.weight.data = child.weight.data.clone()
                setattr(module, name, quant)
    model.is_loaded_in_4bit = True   # what peft reads to pick its bnb LoRA layer


def tiny_lora_model():
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen2 import modeling_qwen2

    config = modeling_qwen2.Qwen2Config(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = modeling_qwen2.Qwen2ForCausalLM(config)
    quantize_linears(model)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    for name, param in model.named_parameters():
        if "lora_B" in name:                 # peft zero-inits B: give the adapter an effect
            torch.nn.init.normal_(param, std=0.2)
    return model.cuda().train()


def inner_model(peft_model):
    return peft_model.base_model.model.model   # the Qwen2Model under peft and ForCausalLM


def batch(right=True):
    lengths = (24, 15, 9)
    seq = max(lengths)
    ids = torch.randint(0, 300, (len(lengths), seq), device="cuda")
    mask = torch.zeros(len(lengths), seq, dtype=torch.long, device="cuda")
    for row, length in enumerate(lengths):
        if right:
            mask[row, :length] = 1
        else:
            mask[row, seq - length:] = 1
    return ids, mask


def run(forward, model, ids, mask, seed):
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.float16):
        return forward(model, input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state


def lora_grads(peft_model):
    return [p.grad.clone() for n, p in peft_model.named_parameters() if "lora_" in n and p.grad is not None]


def pristine_forward(packed_layers, modeling_qwen2):
    """Qwen2Model.forward as transformers shipped it, whether or not the patch is installed."""
    return packed_layers._original_forward or modeling_qwen2.Qwen2Model.forward


def test_packed_matches_padded_with_same_dropout():
    from transformers.models.qwen2 import modeling_qwen2

    from solution import packed_layers

    original = pristine_forward(packed_layers, modeling_qwen2)
    peft_model = tiny_lora_model()
    model = inner_model(peft_model)
    ids, mask = batch(right=True)
    keep = mask.reshape(-1).bool()

    ref = run(original, model, ids, mask, seed=7)
    ref.reshape(-1, 64)[keep].float().pow(2).mean().backward()
    ref_grads = lora_grads(peft_model)
    peft_model.zero_grad(set_to_none=True)

    packed_layers.patch_packed_layers()
    out = run(modeling_qwen2.Qwen2Model.forward, model, ids, mask, seed=7)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    flat_ref, flat_out = ref.reshape(-1, 64).float(), out.reshape(-1, 64).float()
    torch.testing.assert_close(flat_out[keep], flat_ref[keep], rtol=2e-2, atol=2e-2)
    assert (flat_out[~keep] == 0).all(), "pad rows must be zero"
    flat_out[keep].pow(2).mean().backward()
    for got, want in zip(lora_grads(peft_model), ref_grads):
        torch.testing.assert_close(got, want, rtol=5e-2, atol=5e-2)

    other = run(modeling_qwen2.Qwen2Model.forward, model, ids, mask, seed=8)
    diff = (other.reshape(-1, 64).float()[keep] - flat_out[keep]).abs().max()
    assert diff > 1e-3, f"another seed must change the dropout masks, max diff {diff:.2e}"


def test_left_padding_takes_the_original_path():
    from transformers.models.qwen2 import modeling_qwen2

    from solution import packed_layers

    original = pristine_forward(packed_layers, modeling_qwen2)
    model = inner_model(tiny_lora_model())
    ids, mask = batch(right=False)
    ref = run(original, model, ids, mask, seed=3)
    packed_layers.patch_packed_layers()
    out = run(modeling_qwen2.Qwen2Model.forward, model, ids, mask, seed=3)
    assert torch.equal(out, ref)


def test_plan_rejects_what_it_cannot_pack():
    from solution.packed_layers import packing_plan

    model = inner_model(tiny_lora_model())
    ids, mask = batch(right=True)
    with torch.autocast("cuda", dtype=torch.float16):
        assert packing_plan(model, ids, mask, None, None, None, False, {}) is not None
        assert packing_plan(model, ids, mask, None, None, None, True, {}) is None      # cache
        assert packing_plan(model, ids, torch.ones_like(mask), None, None, None, False, {}) is None
        assert packing_plan(model, ids, mask, None, None, None, False, {"foo": 1}) is None
        model.eval()
        assert packing_plan(model, ids, mask, None, None, None, False, {}) is None
    model.train()
    assert packing_plan(model, ids, mask, None, None, None, False, {}) is None          # no autocast
