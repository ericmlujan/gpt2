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


class Transformer(nn.Module):
    MAX_CONTEXT_LENGTH = 4096  # tok
    P_DROPOUT = 0.1

    def __init__(self, n_blocks: int, n_heads: int, d_model: int, d_ff: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.dropout = nn.Dropout(p=self.P_DROPOUT)
        self.encoder_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mha": MultiHeadAttention(self.n_heads, self.d_model),
                        "norm1": LayerNorm(self.d_model),
                        "linear1": nn.Linear(d_model, d_ff),
                        "linear2": nn.Linear(d_ff, d_model),
                        "norm2": LayerNorm(self.d_model),
                    }
                )
                for _ in range(n_blocks)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
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
                for _ in range(n_blocks)
            ]
        )

    def encoder(self, x: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        out = x
        for encoder_block in self.encoder_blocks:
            x_1 = encoder_block["mha"](out, out, out, encoder_mask)
            res_1 = out + self.dropout(x_1)
            x_2 = encoder_block["norm1"](res_1, dim=-1)
            x_3 = F.relu(encoder_block["linear1"](x_2))
            x_4 = encoder_block["linear2"](x_3)
            res_2 = x_2 + self.dropout(x_4)
            out = encoder_block["norm2"](res_2, dim=-1)

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
            d_1 = decoder_block["mha1"](out, out, out, decoder_mask)
            res_1 = out + self.dropout(d_1)
            d_2 = decoder_block["norm1"](res_1, dim=-1)
            # note that q, k are from the outputs of the encoder
            d_3 = decoder_block["mha2"](d_2, encoder_out, encoder_out, encoder_mask)
            res_2 = d_2 + self.dropout(d_3)
            d_4 = decoder_block["norm2"](res_2, dim=-1)
            d_5 = F.relu(decoder_block["linear1"](d_4))
            d_6 = decoder_block["linear2"](d_5)
            res_3 = d_4 + self.dropout(d_6)
            out = decoder_block["norm3"](res_3, dim=-1)

        return out

    def forward(
        self,
        x: torch.Tensor,
        outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.encoder(x, encoder_mask)
        decoder_out = self.decoder(outputs, encoder_out, encoder_mask, decoder_mask)

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


class GPT2Model(nn.Module):
    # This isn't yet a GPT-2, we just want to try doing the actual vanilla transformer
    N_BLOCKS = 6
    N_HEADS = 8
    D_MODEL = 512
    D_FF = 2048
    MAX_CONTEXT_LEN = 256
    P_DROPOUT = 0.1

    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer()
        self.transformer = Transformer(
            self.N_BLOCKS, self.N_HEADS, self.D_MODEL, self.D_FF
        )
        self.w_emb = nn.Parameter(
            torch.empty(self.tokenizer.vocab_size(), self.D_MODEL)
        )
        self.register_buffer(
            "positional_encoding",
            self.transformer.positional_encoding(self.D_MODEL, self.MAX_CONTEXT_LEN),
        )
        self.dropout = nn.Dropout(p=self.P_DROPOUT)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.w_emb, std=0.02)

    def forward(self, x: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        input_embeddings = torch.embedding(self.w_emb, x) * math.sqrt(self.D_MODEL)
        input_embeddings += self.positional_encoding[: input_embeddings.shape[1], :]
        input_embeddings = self.dropout(input_embeddings)

        prev_output_embeddings = torch.embedding(self.w_emb, prev_output) * math.sqrt(
            self.D_MODEL
        )
        prev_output_embeddings += self.positional_encoding[
            : prev_output_embeddings.shape[1], :
        ]
        prev_output_embeddings = self.dropout(prev_output_embeddings)

        # Compute a mask for all padding tokens
        encoder_mask = x == self.tokenizer.pad_token()
        decoder_mask = prev_output == self.tokenizer.pad_token()

        transformer_out = self.transformer.forward(
            input_embeddings, prev_output_embeddings, encoder_mask, decoder_mask
        )

        logits = transformer_out @ self.w_emb.T
        return logits

    def translate(
        self, text: str, stream: bool = False, temperature: Optional[float] = None
    ) -> str:
        encoded_text = self.tokenizer.encode(text)
        encoded_text = self.tokenizer.wrap_eot(encoded_text)
        encoded_text = encoded_text.unsqueeze(dim=0)  # Insert empty batch dim

        encoded_translation = torch.tensor(
            [self.tokenizer.end_of_text_token()]
        ).unsqueeze(dim=0)

        emitted_tok = 0
        last_tok = 0
        while (
            last_tok != self.tokenizer.end_of_text_token()
            and emitted_tok < self.MAX_CONTEXT_LEN
        ):
            with torch.no_grad():
                logits = self.forward(encoded_text, encoded_translation)

                # If a temperature is provided, sample from a categorical distribution with
                # temperature. Otherwise, just plain ol' argmax sampling.
                sampled = torch.tensor(self.tokenizer.end_of_text_token())
                if temperature:
                    logits /= temperature
                    dist = torch.distributions.Categorical(
                        F.softmax(logits, dim=-1)[:, -1]
                    )
                    sampled = dist.sample()
                else:
                    dist = F.softmax(logits, dim=-1)[:, -1]
                    sampled = dist.argmax(dim=-1)

            encoded_translation = torch.concat(
                [encoded_translation, sampled.unsqueeze(dim=0)], dim=1
            )
            emitted_tok += 1
            last_tok = sampled.item()
            if stream:
                print(self.tokenizer.decode(sampled.to("cpu")), end="", flush=True)

        return self.tokenizer.decode(encoded_translation.squeeze(dim=0).to("cpu"))
