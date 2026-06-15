import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_seq_length
    ):

        super().__init__()
        self.embedding = nn.Embedding(
          vocab_size,
          embedding_dim
        )
        self.positional_encoding = PositionalEncoding(
          embedding_dim,
          max_seq_length
        )
        self.encoder = EncoderBlock(
          embedding_dim
        )
    
    def forward(self, x):
        

        print("\nInput:")
        print(x.shape)
        
        embedded = self.embedding(x)
        
        print("\nAfter Embedding:")
        print(embedded.shape)
        
        embedded = self.positional_encoding(
            embedded
        )
        
        print("\nAfter Positional Encoding:")
        print(embedded.shape)
        
        output = self.encoder(
            embedded
        )
        
        print("\nAfter Encoder Block:")
        print(output.shape)
        
        return output
