import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.model import GPT2Tokenizer, Transformer

class TransformerTranslationModel(nn.Module):
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

        # strip leading and final eot tokens
        stripped = encoded_translation[:, 1 : -1]
        return self.tokenizer.decode(stripped.squeeze(dim=0).to("cpu"))