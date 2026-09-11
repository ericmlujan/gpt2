import torch
from gpt2.models.gpt2_model import GPT2Model


def test_gpt2_model():
    model = GPT2Model(64, 128, 8, 2, 256, 0.1)
    text = "hello, world"
    tokenized = model.tokenizer.encode(text).unsqueeze(0)  # add a dummy batch dimension
    seq_length = tokenized.shape[1]
    out = model.forward(tokenized)
    assert out.shape == (1, seq_length, model.tokenizer.vocab_size())
    # check that distributions at all sequence positions sum to 1
    assert torch.allclose(out.sum(dim=-1), torch.ones(1, seq_length))
