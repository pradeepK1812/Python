import torch
import torch.nn as nn


class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        # -----------------------------------
        # Embedding layer
        # Converts word IDs into vectors
        # -----------------------------------

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # -----------------------------------
        # Linear classifier
        # -----------------------------------

        self.fc = nn.Linear(
            embedding_dim,
            1
        )

        # -----------------------------------
        # Probability output
        # -----------------------------------

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # -----------------------------------
        # Convert word IDs to embeddings
        # Shape:
        # [batch, sequence]
        # ->
        # [batch, sequence, embedding_dim]
        # -----------------------------------

        embedded = self.embedding(x)

        # -----------------------------------
        # Average embeddings across sentence
        # dimension
        # -----------------------------------

        pooled = embedded.mean(dim=1)

        # -----------------------------------
        # Linear classification
        # -----------------------------------

        output = self.fc(pooled)

        # -----------------------------------
        # Convert to probability
        # -----------------------------------

        output = self.sigmoid(output)

        return output
