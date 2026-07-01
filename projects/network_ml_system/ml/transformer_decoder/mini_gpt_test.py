import torch

from transformer_decoder.mini_gpt import MiniGPT

model = MiniGPT(
    vocab_size=100,
    embedding_dim=8,
    max_seq_length=10,
    num_layers=3
)

x = torch.randint(
    0,
    100,
    (1,5)
)

output = model(x)

print("\nOutput Shape:")
print(output.shape)
