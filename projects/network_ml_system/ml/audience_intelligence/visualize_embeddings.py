import torch
import matplotlib.pyplot as plt

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
# Load trained model
# -----------------------------------

model.load_state_dict(
    torch.load("sentiment_model.pt")
)

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

embeddings = model.embedding.weight.data

# -----------------------------------
# Use first 2 dimensions for plotting
# -----------------------------------

x = embeddings[:, 0].numpy()

y = embeddings[:, 1].numpy()

# -----------------------------------
# Create plot
# -----------------------------------

plt.figure(figsize=(10, 8))

# -----------------------------------
# Plot each word
# -----------------------------------

for index in range(vocab_size):

    word = index_to_word[index]

    plt.scatter(x[index], y[index])

    plt.text(
        x[index] + 0.02,
        y[index] + 0.02,
        word,
        fontsize=10
    )

# -----------------------------------
# Labels and title
# -----------------------------------

plt.title("Word Embedding Visualization")

plt.xlabel("Embedding Dimension 1")

plt.ylabel("Embedding Dimension 2")

plt.grid(True)

# -----------------------------------
# Save image
# -----------------------------------

plt.savefig(
    "embedding_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nEmbedding visualization saved successfully.")
