import torch
import torch.nn as nn

from transformer_dencoder.masked_multihead_attention import MaskedMultiHeadAttention

class EncoderBlock(nn.Module):

    def __init__(
        self,
        embedding_dim
    ):

        super().__init__()

        # Components will be added gradually

        # -----------------------------------
        # Multi-Head Self Attention
        # -----------------------------------

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=4
        )

        #Normalization after attention
        self.norm1 = nn.LayerNorm(
           embedding_dim
        )

        
        self.ffn = nn.Sequential(

           nn.Linear(
           embedding_dim,
           embedding_dim * 4
           ),

           nn.ReLU(),

           nn.Linear(
            embedding_dim * 4,
            embedding_dim
           )
        )

        self.norm2 = nn.LayerNorm(
          embedding_dim
        )



    #=======================forward pass ============================    
    
    def forward(self, x):

    # -----------------------------------
    # Multi-Head Attention
    # -----------------------------------

        residual = x

        attention_output = self.attention(x)

        x = residual + attention_output

        x = self.norm1(x)

        # -----------------------------------
        # Feed Forward Network
        # -----------------------------------

        residual = x

        ffn_output = self.ffn(x)

        x = residual + ffn_output

        x = self.norm2(x)

        return x
       
