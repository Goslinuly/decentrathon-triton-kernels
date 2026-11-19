"""The background precompilation leaves every kernel of the step fully ready: compiled, its
launcher built and its binary loaded (otherwise the first launch does that inside step 0)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_warm_leaves_kernels_ready(capsys):
    from solution.precompile import _jobs, _warm

    _warm()
    assert "skipped" not in capsys.readouterr().out
    for kernel, args, meta in _jobs():
        compiled = kernel.warmup(*args, grid=(1,), **meta)   # the cached compiled kernel
        assert getattr(compiled, "module", None) is not None, kernel.fn.__name__
