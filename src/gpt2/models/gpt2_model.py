import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt2.model import GPT2Tokenizer, TransformerDecoder


class GPT2Model(nn.Module):
    # Model uses a standard GPT architecture with:
    # - LayerNorm moved to the input of each sub-block
    # - Additional layerNorm added after the last self-attention-block
    # - Residual layers are scaled by 1/sqrt(num residual layers)
    #
    # GPT architecture is the decoder-only transformer with masked self-attention.
    # Model hyperparams: 12 heads, dmodel=768, d_ff=3096, trained on a typical
    # negative log-likelihood loss with Adam and lr incrase.
    #
    # Original GPT was trained for 100 epoch, minibatches were 64 sequences of 512 tokens
    #
    # Dropout rate was 0.1, applied to residuals embedding, and attention.
    #
    # Instead of sinusoidal position embeddings, uses learned position embeddings.
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_blocks: int,
        max_context_length: int,
        p_dropout: float,
    ) -> None:
        super().__init__()

        self.tokenizer = GPT2Tokenizer()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_context_length = max_context_length
        self.p_dropout = p_dropout

        # embedding matrix
        self.w_emb = nn.Parameter(
            torch.empty(self.tokenizer.vocab_size(), self.d_model)
        )
        # positional encoding weights
        self.w_pos = nn.Parameter(torch.empty(self.d_model, self.max_context_length))

        self.decoder_blocks = nn.ParameterList(
            [
                TransformerDecoder(
                    self.n_heads, self.d_model, self.d_ff, self.p_dropout
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize embedding and position encoding weights with normal distribution
        nn.init.normal_(self.w_emb, std=0.02)
        nn.init.normal_(self.w_pos, std=0.02)

    def forward(self, x: torch.Tensor):
        # We don't pad in the decoder-only transformer, so we can set the decoder padding mask to all ones
        # TODO: We can make this optional in the base class too
        pad_mask = torch.ones_like(x, dtype=torch.bool)
        h_0 = torch.embedding(self.w_emb, x) + self.w_pos[:, x.shape[-1]]
        h_dec = h_0
        for block in self.decoder_blocks:
            h_dec = block(h_dec, pad_mask)
        logits = h_dec @ self.w_emb.T
        return F.softmax(logits, dim=-1)[:, -1]
