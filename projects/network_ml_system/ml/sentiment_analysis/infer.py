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

    prediction = model(input_tensor)

# -----------------------------------
# Extract probability
# -----------------------------------

probability = prediction.item()

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
