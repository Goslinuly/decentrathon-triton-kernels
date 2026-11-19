"""Timestamps of the phases of a `soup train` run, written to stderr (the run's train.log).

Analysis only: each line is `[solution] t=<seconds since the process started> <phase>`. It wraps
a few top-level calls (model load, k-bit preparation, PEFT wrapping, trainer init, training,
saving) and prints once when each starts and ends; nothing else changes.
"""

import atexit
import functools
import os
import sys
import time


def _age():
    """Seconds since this process started (Linux /proc), else since import."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return time.perf_counter() - _T0


_T0 = time.perf_counter()


def mark(phase):
    print(f"[solution] t={_age():7.2f} {phase}", file=sys.stderr, flush=True)


def _wrap(owner, name, label, method_type=None):
    original = owner.__dict__.get(name) if isinstance(owner, type) else getattr(owner, name, None)
    if original is None or getattr(original, "_timeline", False):
        return
    func = original.__func__ if isinstance(original, (classmethod, staticmethod)) else original

    @functools.wraps(func)
    def timed(*args, **kwargs):
        mark(f"{label} start")
        try:
            return func(*args, **kwargs)
        finally:
            mark(f"{label} end")

    timed._timeline = True
    if isinstance(original, classmethod):
        timed = classmethod(timed)
    elif isinstance(original, staticmethod):
        timed = staticmethod(timed)
    setattr(owner, name, timed)


def install_timeline():
    mark("apply start")
    atexit.register(mark, "process exit")
    try:
        import peft
        import transformers
        from transformers import Trainer
        from transformers.modeling_utils import PreTrainedModel

        _wrap(PreTrainedModel, "from_pretrained", "from_pretrained")
        _wrap(peft, "prepare_model_for_kbit_training", "prepare_model_for_kbit_training")
        _wrap(peft, "get_peft_model", "get_peft_model")
        _wrap(Trainer, "__init__", "Trainer.__init__")
        _wrap(Trainer, "train", "Trainer.train")
        _wrap(Trainer, "save_model", "Trainer.save_model")
        _wrap(Trainer, "_save_checkpoint", "Trainer._save_checkpoint")
        _wrap(transformers.AutoTokenizer, "from_pretrained", "AutoTokenizer.from_pretrained")
        import datasets
        import trl

        _wrap(trl.SFTTrainer, "__init__", "SFTTrainer.__init__")
        _wrap(trl.SFTTrainer, "_prepare_dataset", "SFTTrainer._prepare_dataset")
        _wrap(datasets.Dataset, "map", "Dataset.map")
        _wrap(datasets, "load_dataset", "datasets.load_dataset")
    except Exception as exc:   # timing must never break a run
        print(f"[solution] timeline not installed: {exc}", file=sys.stderr, flush=True)
