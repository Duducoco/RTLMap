import torch

from text_encoder.config import TextEncoderConfig
from text_encoder.encoder import InstructionEncoder


class _Tokenizer:
    def __call__(self, *_args, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def test_cpu_tokenization_does_not_require_pinned_memory():
    encoder = object.__new__(InstructionEncoder)
    encoder.config = TextEncoderConfig(device="cpu")
    encoder.tokenizer = _Tokenizer()

    tokens = encoder._tokenize_batch(["add x1, x2, x3"])

    assert tokens["input_ids"].tolist() == [[1, 2, 3]]
    assert not tokens["input_ids"].is_pinned()
