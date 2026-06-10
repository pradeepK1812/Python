import math

import torch
import torch.nn as nn


# =========================================================
# POSITIONAL ENCODING
#
# Transformers process all tokens in parallel.
# Unlike RNNs, they do not naturally understand
# token order.
#
# Example:
#
# "dog bites man"
#
# vs
#
# "man bites dog"
#
# Same words.
# Different meaning.
#
# Positional Encoding injects position information
# into token embeddings.
#
# Paper:
# "Attention Is All You Need"
# =========================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embedding_dim,
        max_seq_length
    ):

        super().__init__()

        # =================================================
        # Create position indices
        #
        # Example:
        #
        # [[0]
        #  [1]
        #  [2]
        #  [3]]
        #
        # Shape:
        # [max_seq_length, 1]
        # =================================================

        position = torch.arange(
            max_seq_length
        ).unsqueeze(1)

        # =================================================
        # Create empty positional encoding matrix
        #
        # Shape:
        # [max_seq_length, embedding_dim]
        #
        # Example:
        # [4, 8]
        # =================================================

        pe = torch.zeros(
            max_seq_length,
            embedding_dim
        )

        # =================================================
        # Compute divisor term used by
        # sinusoidal encoding formula.
        #
        # Formula from Transformer paper:
        #
        # PE(pos, 2i)
        # = sin(pos / 10000^(2i/d))
        #
        # PE(pos, 2i+1)
        # = cos(pos / 10000^(2i/d))
        #
        # where:
        #
        # pos = token position
        # i   = embedding dimension index
        # d   = embedding_dim
        # =================================================

        div_term = torch.exp(

            torch.arange(
                0,
                embedding_dim,
                2
            )

            * (

                -math.log(10000.0)

                / embedding_dim
            )
        )

        # =================================================
        # Fill even columns with SIN values
        #
        # Columns:
        # 0, 2, 4, 6 ...
        # =================================================

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # =================================================
        # Fill odd columns with COS values
        #
        # Columns:
        # 1, 3, 5, 7 ...
        # =================================================

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # =================================================
        # Add batch dimension
        #
        # Before:
        # [max_seq_length, embedding_dim]
        #
        # After:
        # [1, max_seq_length, embedding_dim]
        #
        # Example:
        # [1, 4, 8]
        # =================================================

        pe = pe.unsqueeze(0)

        # =================================================
        # Register as buffer
        #
        # Why buffer?
        #
        # - Not trainable
        # - Saved with model
        # - Automatically moved to GPU
        #
        # Not a parameter because we do not
        # want gradient updates.
        # =================================================

        self.register_buffer(
            "pe",
            pe
        )

    # =====================================================
    # FORWARD PASS
    #
    # Input:
    #
    # Embedded tokens
    #
    # Shape:
    # [batch, sequence_length, embedding_dim]
    #
    # Output:
    #
    # Embedded tokens +
    # Positional Encoding
    # =====================================================

    def forward(self, x):

        # ================================================
        # Current sequence length
        #
        # Example:
        #
        # Sentence:
        # "this tutorial is excellent"
        #
        # seq_length = 4
        # ================================================

        seq_length = x.size(1)

        # ================================================
        # Add positional encoding
        #
        # Shape:
        #
        # x:
        # [batch, sequence_length, embedding_dim]
        #
        # pe:
        # [1, sequence_length, embedding_dim]
        #
        # Result:
        #
        # Position-aware embeddings
        # ================================================

        x = x + self.pe[:, :seq_length]

        return x
