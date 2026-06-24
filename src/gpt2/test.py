import pytest
import torch
from mpmath.ctx_mp_python import new

from gpt2.model import GPT2Model, MultiHeadAttention, Transformer


class TestMultiHeadAttention:
    def test_multihead_attention(self):
        n_heads = 8
        d_model = 64
        seq_len = 2
        mha = MultiHeadAttention(n_heads, d_model)
        x = torch.ones(seq_len, d_model)
        y = mha.forward(x, x, x)
        assert x.shape == (seq_len, d_model)

    def test_parameters_registered(self):
        n_heads = 8
        d_model = 64
        mha = MultiHeadAttention(n_heads, d_model)
        assert len(list(mha.parameters())) > 0

    def test_masked_attention(self):
        seq_len = 6
        d_k = 4
        q = torch.randn(seq_len, d_k)
        k = torch.randn(seq_len, d_k)
        v = torch.randn(seq_len, d_k)
        attn = MultiHeadAttention.attention(q, k, v, d_k, causal_mask=True)

        # Now mess with a single position, attention for any tokens before that position shouldn't move
        pos = 3
        k2, v2 = k.clone(), v.clone()
        k2[pos] += 1000
        v2[pos] += 1000
        new_attn = MultiHeadAttention.attention(q, k2, v2, d_k, causal_mask=True)

        assert torch.allclose(attn[:pos], new_attn[:pos])

        # Make sure that for the position that did change, attention did too
        assert not torch.allclose(attn[pos], new_attn[pos])


class TestTransformer:
    def test_forward_dimension(self):
        d_model = 16
        seq_len = 24
        model = Transformer(n_blocks=5, n_heads=2, d_ff=d_model * 4, d_model=d_model)
        input = torch.ones(1, seq_len, d_model)
        prev_output = torch.ones(1, seq_len - 5, d_model)
        output = model.forward(input, prev_output)
        assert output.shape == (1, seq_len - 5, d_model)


class TestGPT2Model:
    def test_forward(self):
        model = GPT2Model()
        x = model.tokenizer.encode("My name is Eric!")
        prev_output = model.tokenizer.encode("stub")
        output = model.forward(x, prev_output)
        assert output.shape == (prev_output.shape[0], model.tokenizer.vocab_size())
        # Make sure logits form a probability distribution along the vocabulary size
        assert torch.allclose(output.sum(dim=-1), torch.ones(prev_output.shape[0]))

    def test_translate(self):
        model = GPT2Model().eval()
        sample_string = "Hello, what is your name?"
        assert model.translate(sample_string) == "Hola, como se llama?"
