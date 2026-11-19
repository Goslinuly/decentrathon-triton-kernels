"""Plug the fused head into Soup's training step.

Two class patches, both looked up at call time:

- `Qwen2ForCausalLM.forward`: during a training step started by the patched trainer below, the
  head and the loss go through `fused_head_loss`; the output carries the loss and no logits.
  Any other call (eval, generation, Soup's tests) takes the original forward.
- `trl.SFTTrainer.compute_loss`: TRL reads `outputs.logits` for `entropy` and
  `mean_token_accuracy`. Our step logs the same columns by TRL's formulas from the kernel's
  per-token results, and leaves them on the trainer as `_soup_chunked_step`, the attribute
  Soup's rewind recorder reads per-row losses from when a step has no logits.

Nothing in Soup's config changes: `stream_loss_chunk`, `use_cut_ce` and the like stay off.
"""

import threading

import torch

from .fused_head import fused_head_loss

# Name of the attribute Soup's rewind recorder (soup_cli/trainer/rewind_hf.py) reads.
REWIND_STEP_ATTR = "_soup_chunked_step"

_state = threading.local()


def _active():
    return getattr(_state, "active", False)


def patch_qwen2_head():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if labels is None or not _active() or logits_to_keep != 0:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        loss, step = fused_head_loss(outputs.last_hidden_state, self.lm_head.weight, labels,
                                     attention_mask, kwargs.get("num_items_in_batch"))
        _state.step = step
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    cls.forward = forward


def _takes_fused_path(trainer, model, inputs):
    if not isinstance(model, torch.nn.Module) or not getattr(model, "training", False):
        return False
    if "labels" not in inputs or "shift_labels" in inputs or "attention_mask" not in inputs:
        return False
    if getattr(trainer, "num_virtual_tokens", 0) or getattr(trainer, "aux_loss_enabled", False):
        return False
    if getattr(trainer.args, "use_liger_kernel", False) or trainer.args.loss_type != "nll":
        return False
    if getattr(trainer, "compute_loss_func", None) is not None:
        return False
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    return any(isinstance(m, Qwen2ForCausalLM) for m in model.modules())


def _log_trl_metrics(trainer, mode, inputs, step):
    """`entropy`, `num_tokens` and `mean_token_accuracy`, in the order and by the formulas of
    `trl.SFTTrainer.compute_loss` (trl 0.29.1), kept as GPU tensors: `deferred.py` reads them in
    `SFTTrainer.log` at the end of the step instead of syncing here, before the backward."""
    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
    trainer._metrics[mode]["entropy"].append(gather(entropy).mean())
    if mode == "train":
        trainer._total_train_tokens = trainer._total_train_tokens + gather(attention_mask.sum()).sum()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    total = gather(step.labelled).sum()
    correct = gather(step.correct).sum()
    accuracy = correct / total.clamp(min=1)
    trainer._metrics[mode]["mean_token_accuracy"].append(
        torch.where(total > 0, accuracy, torch.zeros_like(accuracy)))


def patch_sft_trainer():
    import trl
    from transformers import Trainer

    cls = trl.SFTTrainer
    original = cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _takes_fused_path(self, model, inputs):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        _state.active, _state.step = True, None
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            _state.active = False
        step, _state.step = _state.step, None
        if step is None:
            raise RuntimeError("fused head: the model forward did not take the fused path")
        setattr(self, REWIND_STEP_ATTR, step)
        _log_trl_metrics(self, mode, inputs, step)
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = compute_loss


def patch_head():
    patch_qwen2_head()
    patch_sft_trainer()
