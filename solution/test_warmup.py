"""The GPU warm-up runs every op group and leaves the CUDA RNG state untouched."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_warmup_runs_and_preserves_rng(capsys):
    from solution.warmup import _run

    before = torch.cuda.get_rng_state()
    _run()
    assert torch.equal(before, torch.cuda.get_rng_state())
    # no group may have been skipped: a skip means the op family still warms up at step 0
    assert "warm-up" not in capsys.readouterr().out


def test_warmup_frees_its_memory():
    from solution.warmup import _run

    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    _run()
    # cuBLAS keeps one workspace per stream it has run on (the warm-up uses its own stream),
    # a few tens of MB held by PyTorch's cuBLAS handle pool, not by the warm-up's tensors.
    leftover = torch.cuda.memory_allocated() - before
    assert leftover <= 64 << 20, f"warm-up left {leftover / 2**20:.1f} MB allocated"
