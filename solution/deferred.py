"""Host syncs of the per-step metrics moved from before the backward to the step's log.

TRL's SFTTrainer reads `entropy`, `num_tokens` and `mean_token_accuracy` with `.item()` right
after the forward, and Soup's rewind recorder turns each row's loss into a Python float there
too (`soup_cli/trainer/rewind_hf.py`, `row_losses_from_token_losses` -> `on_micro_batch`): a
dozen GPU syncs per step before the backward is even queued. Our step is CPU-bound, so every
one of them leaves the CPU waiting for the forward to finish instead of queueing the backward.

Here those values stay tensors until `SFTTrainer.log`, which with `logging_steps: 1` runs at the
end of the same step, after the optimizer: they are read there, in the same order, with the same
formulas, so the logged metrics and the rewind.jsonl rows are the same numbers. The harness
synchronizes at every step boundary anyway. Plus `num_parameters` (walked over every module on
every step for the FLOP count) is cached: the parameters of a LoRA run never change.

Plain PyTorch around Soup/TRL; idea from PyTorch's performance tuning guide ("avoid unnecessary
CPU-GPU synchronization").
"""

import torch

_pending_rewind = []   # (RewindState, kwargs) in call order


class LazyValues:
    """A row-wise result kept on the GPU; `len` without a sync, iteration reads it once."""

    __slots__ = ("tensor", "kind", "_values")

    def __init__(self, tensor, kind):
        self.tensor, self.kind, self._values = tensor, kind, None

    def __len__(self):
        return self.tensor.shape[0]

    def __iter__(self):
        if self._values is None:
            self._values = [self.kind(v) for v in self.tensor.tolist()]
        return iter(self._values)


def flush_rewind():
    from soup_cli.trainer import rewind_hf

    original = rewind_hf.RewindState._soup_original_on_micro_batch
    while _pending_rewind:
        state, kwargs = _pending_rewind.pop(0)
        try:
            original(state, **kwargs)
        except Exception as exc:   # as the recorder itself: never raise into training
            state.fail(exc)


def materialize_metrics(trainer):
    metrics = getattr(trainer, "_metrics", None)
    total = getattr(trainer, "_total_train_tokens", None)
    if torch.is_tensor(total):
        trainer._total_train_tokens = int(total.item())
    if not metrics:
        return
    for mode_metrics in metrics.values():
        for key, values in mode_metrics.items():
            for i, value in enumerate(values):
                if torch.is_tensor(value):
                    values[i] = int(value.item()) if key == "num_tokens" else value.item()


def patch_rewind():
    from soup_cli.trainer import rewind_hf

    cls = rewind_hf.RewindState
    if hasattr(cls, "_soup_original_on_micro_batch"):
        return
    cls._soup_original_on_micro_batch = cls.on_micro_batch
    original_summary = cls.summary

    def row_losses_from_token_losses(token_loss, labels):
        # rewind_hf's own formulas, kept on the GPU
        with torch.no_grad():
            tokens = (labels[:, 1:] != -100).sum(dim=1)
            totals = token_loss.float().sum(dim=1).to(tokens.device)
            means = totals / tokens.clamp(min=1).to(totals.dtype)
        return LazyValues(means, float), LazyValues(tokens, int)

    def on_micro_batch(self, *, step, row_loss, row_tokens):
        if isinstance(row_loss, LazyValues):
            _pending_rewind.append((self, dict(step=step, row_loss=row_loss, row_tokens=row_tokens)))
            return None
        flush_rewind()
        return cls._soup_original_on_micro_batch(self, step=step, row_loss=row_loss,
                                                 row_tokens=row_tokens)

    def summary(self):
        flush_rewind()
        return original_summary(self)

    rewind_hf.row_losses_from_token_losses = row_losses_from_token_losses
    cls.on_micro_batch = on_micro_batch
    cls.summary = summary


def patch_log():
    import trl

    cls = trl.SFTTrainer
    original = cls.log
    if getattr(original, "_deferred", False):
        return

    def log(self, logs, start_time=None):
        flush_rewind()
        materialize_metrics(self)
        return original(self, logs, start_time)

    log._deferred = True
    cls.log = log

    # Anything that ends training without a log still writes its rewind rows.
    from transformers import Trainer

    original_train = Trainer.train
    if not getattr(original_train, "_deferred", False):
        def train(self, *args, **kwargs):
            try:
                return original_train(self, *args, **kwargs)
            finally:
                try:
                    flush_rewind()
                    materialize_metrics(self)
                except Exception as exc:   # bookkeeping must never hide how training ended
                    print(f"[solution] deferred metrics at the end of training: "
                          f"{type(exc).__name__}: {exc}", flush=True)

        train._deferred = True
        Trainer.train = train


def patch_num_parameters():
    from transformers.modeling_utils import PreTrainedModel

    original = PreTrainedModel.num_parameters
    if getattr(original, "_cached", False):
        return

    def num_parameters(self, only_trainable=False, exclude_embeddings=False):
        if not self.training:
            return original(self, only_trainable=only_trainable, exclude_embeddings=exclude_embeddings)
        cache = self.__dict__.setdefault("_soup_num_parameters", {})
        key = (only_trainable, exclude_embeddings)
        if key not in cache:
            cache[key] = original(self, only_trainable=only_trainable,
                                  exclude_embeddings=exclude_embeddings)
        return cache[key]

    num_parameters._cached = True
    PreTrainedModel.num_parameters = num_parameters


def patch_deferred():
    patch_rewind()
    patch_log()
    patch_num_parameters()
