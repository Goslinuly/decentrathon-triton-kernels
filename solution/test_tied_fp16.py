"""The fp16 tied embedding: converted only when lossless, output still fp32 and bit-identical,
a trainable or lossy weight left alone. Run on a GPU: `python -m pytest solution -q`."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class TiedToy(torch.nn.Module):
    def __init__(self, vocab=300, dim=64, lossless=True, trainable=False):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            w = torch.randn(vocab, dim)
            if lossless:
                w = w.half().float()           # fp16-representable, as a checkpoint's fp16 weight
            self.embed.weight.copy_(w)
        self.head.weight = self.embed.weight       # tied
        self.embed.weight.requires_grad_(trainable)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def test_lossless_tied_weight_is_shrunk():
    from solution.tied_fp16 import shrink_tied_embedding

    torch.manual_seed(0)
    model = TiedToy().cuda()
    ids = torch.randint(0, 300, (3, 17), device="cuda")
    before = model.embed(ids)
    assert shrink_tied_embedding(model) == 1
    assert model.embed.weight.dtype == torch.float16
    assert model.head.weight is model.embed.weight
    after = model.embed(ids)
    assert after.dtype == torch.float32
    assert torch.equal(after, before)


@pytest.mark.parametrize("lossless,trainable", [(False, False), (True, True)])
def test_lossy_or_trainable_weight_is_left_alone(lossless, trainable):
    from solution.tied_fp16 import shrink_tied_embedding

    torch.manual_seed(0)
    model = TiedToy(lossless=lossless, trainable=trainable).cuda()
    assert shrink_tied_embedding(model) == 0
    assert model.embed.weight.dtype == torch.float32


def test_non_module_is_ignored():
    from solution.tied_fp16 import shrink_tied_embedding

    assert shrink_tied_embedding(object()) == 0
