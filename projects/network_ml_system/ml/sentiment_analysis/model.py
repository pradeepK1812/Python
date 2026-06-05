import torch
import torch.nn as nn


class SentimentModel(nn.Module):

    def __init__(self, vocab_size, max_seq_length , embedding_dim):

        super().__init__()
        
        # -----------------------------------
        # Multi-Head Attention Parameters
        # -----------------------------------

        self.num_heads = 4

        self.embedding_dim = embedding_dim

        self.head_dim = (
           embedding_dim // self.num_heads
        )
        # -----------------------------------
        # Embedding layer
        # Converts word IDs into vectors
        # -----------------------------------

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # -----------------------------------
        # Positional Embedding
        #
        # Learns sequence position information
        # -----------------------------------

        self.position_embedding = nn.Embedding(
         max_seq_length,
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
        # Output projection
        #
        # Combines all head outputs
        # into final contextual representation
        # -----------------------------------

        self.out = nn.Linear(
         embedding_dim,
         embedding_dim
        )
        # -----------------------------------
        # FeedForward Network (FFN)
        #
        # First Linear:
        # expands representation capacity
        #
        # Second Linear:
        # compresses back to embedding size
        # -----------------------------------

        self.ffn = nn.Sequential(

        nn.Linear(
         embedding_dim,
         embedding_dim * 2
        ),

        nn.ReLU(),

        nn.Linear(
         embedding_dim * 2,
         embedding_dim
        )
        )
        # -----------------------------------
        # Layer Normalization
        #
        # Stabilizes representation magnitudes
        # across embedding dimensions
        # -----------------------------------

        self.layer_norm = nn.LayerNorm(
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

      embedded = self.embedding(x) #after addition of positional embedding now it needs to be moved after positional embedding
      
      # -----------------------------------
      # Create position indices
      # -----------------------------------

      positions = torch.arange(
        x.shape[1]
      ).unsqueeze(0)
      #Get positional embeddings 
      position_embeddings = self.position_embedding(
        positions
      )
      #Now embedding is embedded + positional embedding
      embedded = embedded + position_embeddings
    # -----------------------------------
    # Generate Q/K/V projections
    # -----------------------------------

      Q = self.query(embedded)

      K = self.key(embedded)

      V = self.value(embedded)

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
      #attention_scores = torch.matmul(
      # Q,
      # K.transpose(1, 2)
      #) / (Q.shape[-1] ** 0.5)

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
      
      # -----------------------------------
      # FeedForward refinement
      #
      # Refines contextual token
      # representations
      # -----------------------------------

      #refined = self.ffn(attended)
      refined = attended + self.ffn(attended) #added attended also so that old info does not get lost and it also gets added
      normalized = self.layer_norm(refined) #Layer normalization added after FFN
      # -----------------------------------
      # Aggregate sequence information
      #
      # Mean pooling across sequence
      # dimension
      # -----------------------------------

      #pooled = attended.mean(dim=1)
      #pooled = refined.mean(dim=1) #now take the mean of refined instead of attneded
      pooled = normalized.mean(dim=1) #now take the mean of refined instead of attneded

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
