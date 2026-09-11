import math
from typing import Optional

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_DTYPE = torch.int32


class GPT2Tokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self._enc.encode(text), dtype=TOKEN_DTYPE)

    def decode(self, tok: torch.Tensor) -> str:
        return self._enc.decode(tok.tolist())

    def vocab_size(self) -> int:
        return self._enc.max_token_value + 2

    def end_of_text_token(self) -> int:
        return self._enc.eot_token

    def pad_token(self) -> int:
        return self._enc.max_token_value + 1

    def shift_right(self, seq: torch.Tensor) -> torch.Tensor:
        # Lop off the first token, append the EOT token
        return torch.concat(
            [seq[1:], torch.tensor(self.end_of_text_token()).unsqueeze(dim=0)]
        )

    def prepend_eot(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.concat(
            [torch.tensor(self.end_of_text_token()).unsqueeze(dim=0), seq]
        )

    def wrap_eot(self, seq: torch.Tensor) -> torch.Tensor:
        eot_tensor = torch.tensor(self.end_of_text_token()).unsqueeze(dim=0)
        return torch.concat([eot_tensor, seq, eot_tensor])


# Hand-implementation of the Transformer architecture from the "Attention is All You Need Paper"
class MultiHeadAttention(nn.Module):
    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        d_k: int,
        causal_mask: bool,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        # If a causal mask mode is defined, a given position shouldn't be able to attend to positions
        # in the future. We mask above the diagonal (diagonal=1), since positions are allowed to attend to themselves.
        # The masking works because putting -inf into softmax pulls those values to 0.
        if causal_mask:
            mask = torch.ones_like(y, dtype=torch.bool).triu(diagonal=1)
            y = y.masked_fill(mask, value=-float("inf"))

        # Mask out any positions that are requested to be masked by the caller.
        # This is done to mask out padding tokens that shouldn't be attended to.
        if attn_mask is not None:
            y = y.masked_fill(attn_mask[:, None, :], value=-float("inf"))

        return F.softmax(y, dim=-1) @ v

    def __init__(self, n_heads: int, d_model: int, causal_mask: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be evenly divisible by n_heads ({n_heads})"
            )

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.causal_mask = causal_mask

        self.w_q = nn.ParameterList(
            [torch.empty(d_model, self.d_k) for _ in range(n_heads)]
        )
        self.w_k = nn.ParameterList(
            [torch.empty(d_model, self.d_k) for _ in range(n_heads)]
        )
        self.w_v = nn.ParameterList(
            [torch.empty(d_model, self.d_v) for _ in range(n_heads)]
        )

        self.w_o = nn.Parameter(torch.empty(d_model, d_model))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_o)

        for i in range(self.n_heads):
            nn.init.xavier_uniform_(self.w_q[i])
            nn.init.xavier_uniform_(self.w_k[i])
            nn.init.xavier_uniform_(self.w_v[i])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        combined = torch.concat(
            [
                self.attention(
                    q @ self.w_q[i],
                    k @ self.w_k[i],
                    v @ self.w_v[i],
                    self.d_k,
                    causal_mask=self.causal_mask,
                    attn_mask=attn_mask,
                )
                for i in range(self.n_heads)
            ],
            dim=-1,
        )

        return combined @ self.w_o


class LayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.EPS = 1e-6
        self.d_model = d_model
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.gain = nn.Parameter(torch.ones(self.d_model))

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        mean = torch.mean(x, dim, keepdim=True)
        var = torch.var(x, dim, keepdim=True)
        return (self.gain / torch.sqrt(var + self.EPS)) * (x - mean) + self.bias


class TransformerDecoder(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, p_dropout: float):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.dropout = nn.Dropout(p_dropout)

        self.nn = nn.ModuleDict(
            {
                "mha1": MultiHeadAttention(
                    self.n_heads, self.d_model, causal_mask=True
                ),
                "norm1": LayerNorm(self.d_model),
                "mha2": MultiHeadAttention(self.n_heads, self.d_model),
                "norm2": LayerNorm(self.d_model),
                "linear1": nn.Linear(d_model, d_ff),
                "linear2": nn.Linear(d_ff, d_model),
                "norm3": LayerNorm(self.d_model),
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        encoder_out: torch.Tensor | None = None,
        encoder_mask: torch.Tensor | None = None,
    ):
        if (encoder_out is not None and encoder_mask is None) or (
            encoder_mask is not None and encoder_out is None
        ):
            raise ValueError(
                "if using cross-attention, must specify both the encoder's output and the encoder mask"
            )

        d_1 = self.nn["mha1"](x, x, x, mask)
        res_1 = x + self.dropout(d_1)
        d_2 = self.nn["norm1"](res_1, dim=-1)
        # if using cross-attention, q and k are from the outputs of the encoder and the mask should agree with the mask used for the encoder
        # else, skip that block and go straight to the feedforward network
        if encoder_out is not None:
            d_3 = self.nn["mha2"](d_2, encoder_out, encoder_out, encoder_mask)
            res_2 = d_2 + self.dropout(d_3)
            d_4 = self.nn["norm2"](res_2, dim=-1)
        else:
            d_4 = d_2
        d_5 = F.relu(self.nn["linear1"](d_4))
        d_6 = self.nn["linear2"](d_5)
        res_3 = d_4 + self.dropout(d_6)
        out = self.nn["norm3"](res_3, dim=-1)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, p_dropout: float):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.dropout = nn.Dropout(p_dropout)

        self.nn = nn.ModuleDict(
            {
                "mha": MultiHeadAttention(self.n_heads, self.d_model),
                "norm1": LayerNorm(self.d_model),
                "linear1": nn.Linear(d_model, d_ff),
                "linear2": nn.Linear(d_ff, d_model),
                "norm2": LayerNorm(self.d_model),
            }
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_1 = self.nn["mha"](x, x, x, mask)
        res_1 = x + self.dropout(x_1)
        x_2 = self.nn["norm1"](res_1, dim=-1)
        x_3 = F.relu(self.nn["linear1"](x_2))
        x_4 = self.nn["linear2"](x_3)
        res_2 = x_2 + self.dropout(x_4)
        out = self.nn["norm2"](res_2, dim=-1)
        return out


class Transformer(nn.Module):
    MAX_CONTEXT_LENGTH = 4096  # tok
    P_DROPOUT = 0.1

    def __init__(self, n_blocks: int, n_heads: int, d_model: int, d_ff: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoder(
                    self.n_heads, self.d_model, self.d_ff, self.P_DROPOUT
                )
                for _ in range(n_blocks)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoder(
                    self.n_heads, self.d_model, self.d_ff, self.P_DROPOUT
                )
                for _ in range(n_blocks)
            ]
        )

    def encoder(self, x: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        out = x
        for encoder_block in self.encoder_blocks:
            out = encoder_block(out, encoder_mask)
        return out

    def decoder(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = x
        for decoder_block in self.decoder_blocks:
            out = decoder_block(out, decoder_mask, encoder_out, encoder_mask)
        return out

    def forward(
        self,
        x: torch.Tensor,
        outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.encoder(x, encoder_mask)
        print(f"Encoder out: {encoder_out}")
        decoder_out = self.decoder(outputs, encoder_out, encoder_mask, decoder_mask)
        print(f"Decoder out: {decoder_out}")

        return decoder_out

    @staticmethod
    def positional_encoding(
        d_model: int, context_len: int = MAX_CONTEXT_LENGTH
    ) -> torch.Tensor:
        upper = math.ceil(d_model / 2)
        positions = torch.arange(context_len)
        # freqs[i] = 1 / 10000^(2i/d_model), one frequency per sin/cos pair
        freqs = 1.0 / torch.pow(10_000, 2 * torch.arange(upper) / d_model)
        # outer product gives (context_len, upper) where cell [p, i] = pos / denom(i)
        combined = torch.outer(positions, freqs)
        sines = torch.sin(combined)
        cosines = torch.cos(combined)
        # stack -> (context_len, 2, upper), transpose -> (context_len, upper, 2),
        # flatten last two dims interleaves sin/cos pairs: [s0, c0, s1, c1, ...]
        return torch.flatten(
            torch.stack([sines, cosines], dim=1).transpose(1, 2), start_dim=1
        )[:, :d_model]
