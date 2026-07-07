import torch
import torch.nn as nn


from ml.transformer_encoder.positional_encoding import PositionalEncoding
from ml.transformer_decoder.decoder_block import DecoderBlock

class TransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_seq_length,
        num_layers # added to support multi-layer encoder
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
       # self.encoder = EncoderBlock(embedding_dim) 
       #single block removed and multi-layer added below

        self.layers = nn.ModuleList(
        [
            DecoderBlock(
                embedding_dim
            )
            for _ in range(num_layers)
        ]
)
    
    def forward(self, x):
        

        #print("\nInput:")
        #print(x.shape)
        
        embedded = self.embedding(x)
        
        #print("\nAfter Embedding:")
        #print(embedded.shape)
        
        embedded = self.positional_encoding(
            embedded
        )
        
        #print("\nAfter Positional Encoding:")
        #print(embedded.shape)
        
       # output = self.encoder( embedded)
       #single encoding block commented and multi-layer encoding block added 

        x = embedded
        layer_num = 1 # to print the layers

        for layer in self.layers:

            x = layer(x)
            #to print the layer info for debug purpose
            """  print(
                     f"\nAfter Decoder Layer {layer_num}:"
                 )

            print(x.shape)"""

            layer_num += 1


        return x
        


                    
