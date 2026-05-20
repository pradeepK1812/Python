import torch

from model import SentimentModel
from vocabulary import word_to_index

# -----------------------------------
# Model configuration
# -----------------------------------

vocab_size = len(word_to_index)

embedding_dim = 64
max_seq_length = 15
# -----------------------------------
# Create model
# -----------------------------------

model = SentimentModel(
    vocab_size,
    max_seq_length,
    embedding_dim
)

# -----------------------------------
# Load trained weights
# -----------------------------------

model.load_state_dict(
    torch.load("sentiment_model.pt")
)

# -----------------------------------
# Set model to inference mode
# -----------------------------------

model.eval()

# -----------------------------------
# Convert sentence into token IDs
# -----------------------------------

def tokenize(text):

    words = text.lower().split()

    token_ids = []

    for word in words:

        # Ignore unknown words for now
        if word in word_to_index:

            token_ids.append(
                word_to_index[word]
            )

    return token_ids

# -----------------------------------
# Input sentence
# -----------------------------------

text = "this tutorial is excellent"

# -----------------------------------
# Convert to token IDs
# -----------------------------------

tokens = tokenize(text)

print("\nToken IDs:\n")

print(tokens)

# -----------------------------------
# Convert to tensor
# -----------------------------------

input_tensor = torch.tensor(
    [tokens]
)

# -----------------------------------
# Run inference
# -----------------------------------

with torch.no_grad():

   # prediction = model(input_tensor)
    prediction, attention_weights = model(input_tensor)

# -----------------------------------
# Extract probability
# -----------------------------------

probability = prediction.item()
# -----------------------------------
# Display attention weights
# -----------------------------------

words = text.lower().split()

weights = attention_weights.squeeze().tolist()

attention = attention_weights.squeeze()

print("\nAttention Matrix:\n")

for i, word in enumerate(words):

    print(f"\n{word} attends to:\n")

    for j, target_word in enumerate(words):

        score = attention[i][j].item()

        print(
            f"  {target_word} -> {score:.4f}"
        )
        
"""print("\nAttention Weights:\n")

for word, weight in zip(words, weights):

    print(
        f"{word} -> {weight:.4f}"
    )"""
# -----------------------------------
# Determine sentiment
# -----------------------------------

if probability >= 0.5:

    sentiment = "positive"

else:

    sentiment = "negative"

# -----------------------------------
# Confidence percentage
# -----------------------------------

confidence = f"{probability * 100:.0f}%"

# -----------------------------------
# Print result
# -----------------------------------

print("\nPrediction Result:\n")

print({
    "sentiment": sentiment,
    "confidence": confidence
})
