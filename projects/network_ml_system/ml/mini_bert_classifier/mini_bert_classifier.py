import torch
import torch.nn as nn
import sys
import os


# from folder.file import Class
from ml.transformer_encoder.transformer_encoder import TransformerEncoder

class MiniBERTClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_seq_length,
        num_layers,
        num_classes
    ):

        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

            encoder_output = self.encoder(x)
            """ print(
                     f"\nAfter Encoder output:"
                 )

            print(encoder_output.shape)"""


            cls_output = encoder_output[:, 0, :]
            """print(
                     f"\nAfter CLS output :"
                 )

            print(cls_output.shape)"""


            logits = self.classifier(
                cls_output
            )

            """print(
                     f"\nLogits :"
                 )

            print(logits)"""

            return logits
