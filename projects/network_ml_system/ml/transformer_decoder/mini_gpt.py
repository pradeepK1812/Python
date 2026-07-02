import torch
import torch.nn as nn

from transformer_decoder.transformer_decoder import (
    TransformerDecoder
)

class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_seq_length,
        num_layers
    ):

        super().__init__()
        #decoder instance
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            num_layers=num_layers
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.decoder(x)

        logits = self.lm_head(x)
        
        """ print("\nAfter Transformer Decoder:")
        print(x.shape)

        print("\nVocabulary Logits:")
        print(logits.shape)"""
       
        return logits
