import torch

from transformer_encoder import TransformerEncoder

# -----------------------------------
# Configuration
# -----------------------------------

vocab_size = 100
embedding_dim = 8
max_seq_length = 10

# -----------------------------------
# Create encoder
# -----------------------------------

encoder = TransformerEncoder(
    vocab_size,
    embedding_dim,
    max_seq_length
)

# -----------------------------------
# Dummy token IDs
#
# Shape:
# [batch, sequence]
# -----------------------------------

x = torch.randint(
    0,
    vocab_size,
    (1, 4)
)

print("\nInput Tokens:\n")
print(x)

print("\nInput Shape:\n")
print(x.shape)

# -----------------------------------
# Run encoder
# -----------------------------------

output = encoder(x)

print("\nOutput Shape:\n")
print(output.shape)
