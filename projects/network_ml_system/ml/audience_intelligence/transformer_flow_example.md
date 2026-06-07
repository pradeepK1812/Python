# =========================================================
# QKV SELF-ATTENTION PIPELINE WITH TENSOR DIMENSIONS
# =========================================================

# Assumptions:
#
# batch_size      = 1
# sequence_length = 4
# embedding_dim   = 8
#
# Example sentence:
# ["this", "tutorial", "is", "excellent"]

# =========================================================
# STEP 1 — INPUT TOKEN IDS
# =========================================================

# Shape:
# [batch, sequence]

x.shape

# Example:
# [1, 4]

# Meaning:
# 1 sentence
# 4 tokens


# =========================================================
# STEP 2 — EMBEDDING LOOKUP
# =========================================================

embedded = self.embedding(x)

# Shape:
# [batch, sequence, embedding_dim]

embedded.shape

# Example:
# [1, 4, 8]

# Meaning:
# 1 batch
# 4 token vectors
# each token represented by 8 dimensions


# =========================================================
# STEP 3 — QUERY / KEY / VALUE PROJECTIONS
# =========================================================

Q = self.query(embedded)

K = self.key(embedded)

V = self.value(embedded)

# Shape of Q:
# [batch, sequence, embedding_dim]

Q.shape

# Example:
# [1, 4, 8]


# Shape of K:
# [batch, sequence, embedding_dim]

K.shape

# Example:
# [1, 4, 8]


# Shape of V:
# [batch, sequence, embedding_dim]

V.shape

# Example:
# [1, 4, 8]


# Meaning:
# Same token embeddings transformed into:
#
# Query space
# Key space
# Value space


# =========================================================
# STEP 4 — TRANSPOSE KEY MATRIX
# =========================================================

K_transposed = K.transpose(1, 2)

# Original K shape:
# [1, 4, 8]

# After transpose(1,2):
# [1, 8, 4]

K_transposed.shape

# Meaning:
# Converts:
#
# [sequence x embedding_dim]
#
# into:
#
# [embedding_dim x sequence]
#
# required for matrix multiplication


# =========================================================
# STEP 5 — COMPUTE ATTENTION SCORES
# =========================================================

attention_scores = torch.matmul(
    Q,
    K_transposed
)

# Shape:
# [batch, sequence, sequence]

attention_scores.shape

# Example:
# [1, 4, 4]

# Meaning:
# Every token compared with every token


# =========================================================
# STEP 6 — APPLY SOFTMAX
# =========================================================

attention_weights = torch.softmax(
    attention_scores,
    dim=2
)

# Shape:
# [batch, sequence, sequence]

attention_weights.shape

# Example:
# [1, 4, 4]

# Meaning:
# Attention probabilities
#
# Each row sums to 1
#
# Represents:
# how much each token attends
# to every other token


# =========================================================
# STEP 7 — APPLY ATTENTION TO VALUES
# =========================================================

attended = torch.matmul(
    attention_weights,
    V
)

# Shapes:
#
# attention_weights:
# [1, 4, 4]
#
# V:
# [1, 4, 8]

# Result:
# [1, 4, 8]

attended.shape

# Meaning:
# Context-aware token representations
#
# Each token representation now
# contains contextual information
# from other tokens


# =========================================================
# STEP 8 — POOLING
# =========================================================

pooled = attended.mean(dim=1)

# Shape:
# [batch, embedding_dim]

pooled.shape

# Example:
# [1, 8]

# Meaning:
# Single sentence representation


# =========================================================
# STEP 9 — LINEAR CLASSIFIER
# =========================================================

output = self.fc(pooled)

# Shape:
# [batch, 1]

output.shape

# Example:
# [1, 1]

# Meaning:
# Raw prediction score


# =========================================================
# STEP 10 — SIGMOID
# =========================================================

output = self.sigmoid(output)

# Shape:
# [batch, 1]

output.shape

# Example:
# [1, 1]

# Meaning:
# Final probability
#
# Example:
# 0.92 = positive sentiment
#######################################################################################
