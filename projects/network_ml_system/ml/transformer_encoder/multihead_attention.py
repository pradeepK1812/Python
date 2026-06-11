# multi_head_attention.py

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):

        super().__init__()

        if embedding_dim % num_heads != 0:

            raise ValueError(
               "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim

        self.num_heads = num_heads

        self.head_dim = (
            embedding_dim // num_heads
        )

         # -----------------------------------
        # Query projection
        # Learns what information
        # each token is searching for
        # -----------------------------------

        self.query = nn.Linear(
        embedding_dim,
        embedding_dim
        )

        # -----------------------------------
        # Key projection
        # Learns what information
        # each token contains
        # -----------------------------------

        self.key = nn.Linear(
        embedding_dim,
        embedding_dim
        )

        # -----------------------------------
        # Value projection
        # Learns what information
        # each token contributes
        # -----------------------------------

        self.value = nn.Linear(
        embedding_dim,
        embedding_dim
        )

        # -----------------------------------
        # Output projection
        #
        # Combines all head outputs
        # into final contextual representation
        # -----------------------------------

        self.out = nn.Linear(
         embedding_dim,
         embedding_dim
        )

    def forward(self, x):

       
    # -----------------------------------
    # Input:
    #
    # x shape:
    # [batch, sequence, embedding_dim]
    #
    # Already contains:
    # token embeddings +
    # positional encoding
    # -----------------------------------

      # -----------------------------------
      # Generate Q/K/V projections
      # -----------------------------------

      Q = self.query(x)

      K = self.key(x)

      V = self.value(x)

      # -----------------------------------
      # Split embeddings into multiple heads
      #
      # [batch, seq, embedding_dim]
      # ->
      # [batch, seq, num_heads, head_dim]
      # -----------------------------------

      Q = Q.view(
        Q.shape[0],
        Q.shape[1],
        self.num_heads,
        self.head_dim
      )

      K = K.view(
        K.shape[0],
        K.shape[1],
        self.num_heads,
        self.head_dim
      )

      V = V.view(
        V.shape[0],
        V.shape[1],
        self.num_heads,
        self.head_dim
      )

      # -----------------------------------
      # Move heads dimension before sequence
      #
      # [batch, seq, heads, head_dim]
      # ->
      # [batch, heads, seq, head_dim]
      # -----------------------------------

      Q = Q.transpose(1, 2)

      K = K.transpose(1, 2)

      V = V.transpose(1, 2)

     
      # -----------------------------------
      # Multi-head scaled attention
      #
      # Attention computed independently
      # for each head
      # -----------------------------------

      attention_scores = torch.matmul(
       Q,
       K.transpose(-2, -1)
      ) / (self.head_dim ** 0.5)
    # -----------------------------------
    # Convert similarity scores into
    # probability distribution
    #
    # Softmax makes each row sum to 1
    #
    # Higher score =
    # stronger attention relationship
    # -----------------------------------

      attention_weights = torch.softmax(
        attention_scores,
        dim=-1
      )

    # -----------------------------------
    # Apply self-attention
    #
    # attention_weights:
    # [batch, sequence, sequence]
    #
    # embedded:
    # [batch, sequence, embedding_dim]
    #
    # Result:
    # [batch, sequence, embedding_dim]
    #
    # Each token representation now
    # becomes context-aware using
    # other tokens
    # -----------------------------------

      #attended = torch.matmul(
       # attention_weights,
        #embedded
      #)
      attended = torch.matmul(
       attention_weights,
       V
      )

      # -----------------------------------
      # Move sequence dimension back
      #
      # [batch, heads, seq, head_dim]
      # ->
      # [batch, seq, heads, head_dim]
      # -----------------------------------

      attended = attended.transpose(1, 2)

      # -----------------------------------
      # Merge all heads together
      #
      # [batch, seq, heads, head_dim]
      # ->
      # [batch, seq, embedding_dim]
      # -----------------------------------

      attended = attended.contiguous().view(
       attended.shape[0],
       attended.shape[1],
       self.embedding_dim
      )

      # -----------------------------------
      # Mix information from all heads
      # -----------------------------------

      attended = self.out(attended)

      #Return attended now
      return attended
