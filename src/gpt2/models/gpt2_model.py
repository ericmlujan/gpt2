import torch.nn as nn


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
    #
    # TODO:
    # - Need to factor out the decoder block from the Transformer class
    def __init__(self):
        super().__init__()

    def forward():
        pass

    def reset_parameters():
        pass
