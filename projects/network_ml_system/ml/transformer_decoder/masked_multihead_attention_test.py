import torch

from transformer_decoder.masked_multihead_attention import (
    MaskedMultiHeadAttention
)

attention = MaskedMultiHeadAttention(
    embedding_dim=8,
    num_heads=2
)

x = torch.randn(
    1,
    5,
    8
)

output = attention(x)

print("Input:")
print(x.shape)

print("\nOutput:")
print(output.shape)
