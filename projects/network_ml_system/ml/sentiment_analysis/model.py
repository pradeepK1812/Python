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
    # Convert token IDs into embeddings
    #
    # Shape:
    # [batch, sequence]
    #
    # becomes:
    #
    # [batch, sequence, embedding_dim]
    # -----------------------------------

      embedded = self.embedding(x)
      
    # -----------------------------------
    # Generate Q/K/V projections
    # -----------------------------------

      Q = self.query(embedded)

      K = self.key(embedded)

      V = self.value(embedded)
    # -----------------------------------
    # Compute token-to-token similarity
    #
    # embedded:
    # [batch, sequence, embedding_dim]
    #
    # embedded.transpose(1, 2):
    # [batch, embedding_dim, sequence]
    #
    # Result:
    # [batch, sequence, sequence]
    #
    # Each token compares itself
    # with every other token
    # -----------------------------------

      #attention_scores = torch.matmul(
       # embedded,
        #embedded.transpose(1, 2)
      #)
      # added QKTV/Square root of DK (key dimension)
      attention_scores = torch.matmul(
       Q,
       K.transpose(1, 2)
      ) / (Q.shape[-1] ** 0.5)
 
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
        dim=2
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
    # Aggregate sequence information
    #
    # Mean pooling across sequence
    # dimension
    # -----------------------------------

      pooled = attended.mean(dim=1)

    # -----------------------------------
    # Linear classification layer
    # -----------------------------------

      output = self.fc(pooled)

    # -----------------------------------
    # Convert logits into probability
    # -----------------------------------

      output = self.sigmoid(output)

    # -----------------------------------
    # Return prediction and attention
    # weights for inspection
    # -----------------------------------

      return output, attention_weights


##########################################################

"""    def forward(self, x):

    # -----------------------------------
    # Convert token IDs into embeddings
    # Shape:
    # [batch, sequence]
    # ->
    # [batch, sequence, embedding_dim]
    # -----------------------------------

       embedded = self.embedding(x)

    # -----------------------------------
    # Compute attention scores
    # by averaging embedding dimensions
    # -----------------------------------

       attention_scores = embedded.mean(dim=2)

    # -----------------------------------
    # Convert scores into probabilities
    # across sequence dimension
    # -----------------------------------

       attention_weights = torch.softmax(
       attention_scores,
       dim=1
       )

    # -----------------------------------
    # Expand weights for multiplication
    # Shape:
    # [batch, sequence]
    # ->
    # [batch, sequence, 1]
    # -----------------------------------

       attention_weights = attention_weights.unsqueeze(2)

    # -----------------------------------
    # Apply attention weights
    # -----------------------------------

       weighted_embeddings = (
       embedded * attention_weights
       )

    # -----------------------------------
    # Aggregate weighted embeddings
    # -----------------------------------

       pooled = weighted_embeddings.sum(dim=1)

    # -----------------------------------
    # Linear classification
    # -----------------------------------

       output = self.fc(pooled)

    # -----------------------------------
    # Convert to probability
    # -----------------------------------

       output = self.sigmoid(output)

       #return output
       return output, attention_weights

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

        return output"""
