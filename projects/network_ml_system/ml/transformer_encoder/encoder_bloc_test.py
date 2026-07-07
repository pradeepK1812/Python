# encoder_block_test.py

import torch

from ml.transformer_encoder.encoder_block import EncoderBlock

block = EncoderBlock(
    embedding_dim=8
)

x = torch.randn(
    1,
    4,
    8
)

output = block(x)

print(x.shape)
print(output.shape)
