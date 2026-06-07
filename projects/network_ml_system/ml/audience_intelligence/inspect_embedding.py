import torch

from model import SentimentModel
from vocabulary import word_to_index

# -----------------------------------
# Model configuration
# -----------------------------------

vocab_size = len(word_to_index)

embedding_dim = 8

# -----------------------------------
# Create model
# -----------------------------------

model = SentimentModel(
    vocab_size,
    embedding_dim
)

# -----------------------------------
# Load trained weights
# -----------------------------------

model.load_state_dict(
    torch.load("sentiment_model.pt")
)

# -----------------------------------
# Set inference mode
# -----------------------------------

model.eval()

# -----------------------------------
# Reverse vocabulary mapping
# -----------------------------------

index_to_word = {
    index: word
    for word, index in word_to_index.items()
}

# -----------------------------------
# Extract embedding matrix
# -----------------------------------

embedding_weights = model.embedding.weight.data

# -----------------------------------
# Print embeddings
# -----------------------------------

print("\nLearned Embeddings:\n")

for index in range(vocab_size):

    word = index_to_word[index]

    vector = embedding_weights[index]

    print(f"{word} ->\n{vector}\n")
