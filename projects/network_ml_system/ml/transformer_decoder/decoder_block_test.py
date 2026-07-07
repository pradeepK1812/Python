import torch

from ml.transformer_decoder.decoder_block import (
    DecoderBlock
)

decoder = DecoderBlock(
    embedding_dim=8
)

x = torch.randn(
    1,
    5,
    8
)

output = decoder(x)

print("Input:")
print(x.shape)

print("\nOutput:")
print(output.shape)
