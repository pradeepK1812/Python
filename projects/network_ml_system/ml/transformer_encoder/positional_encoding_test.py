import math
import torch
import torch.nn as nn
from ml.transformer_encoder.positional_encoding import PositionalEncoding

embedding_dim = 8
max_seq_length = 6

pe = PositionalEncoding(
    embedding_dim,
    max_seq_length
)
print("printing the shape and size of PE")
print(pe.pe.shape)
print("\nPosition 0:")
print(pe.pe[0,0])
print("\nPosition 1:")
print(pe.pe[0,1])
print("\nPosition 2:")
print(pe.pe[0,2])
#print(pe.pe)
