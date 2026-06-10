import torch

from model import SentimentModel
from vocabulary import word_to_index


# -----------------------------------
# Emotion label decoder
# -----------------------------------

index_to_label = {

    0: "appreciation",
    1: "confusion",
    2: "curiosity",
    3: "frustration",
    4: "excitement",
    5: "boredom"
}
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

   # token_ids = [] #As now we have added the CLS tocken at the start
    token_ids = [
      word_to_index["[CLS]"]
    ]

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

#text = "this tutorial was amazing but still confusing"
#text = "this tutorial is amazing"
text = "very helpful tutorial"
print(f"\nInput sentence is :{text} \n")
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
# Convert logits into probabilities
# -----------------------------------

probabilities = torch.softmax(
    prediction,
    dim=-1
)

# -----------------------------------
# Get top-k emotions
# -----------------------------------

top_probs, top_indices = torch.topk(
    probabilities,
    k=3,
    dim=-1
)

print("\nTop Emotion Analysis:\n")

meaningful_emotions = []

for i in range(3):

    class_index = top_indices[0][i].item()

    emotion = index_to_label[
        class_index
    ]

    confidence = top_probs[0][i].item()

    print(
        f"{emotion} -> "
        f"{confidence * 100:.2f}%"
    )
    
    # -----------------------------------
    # Keep only meaningful emotions
    # -----------------------------------

    if confidence >= 0.30:

        meaningful_emotions.append(
            (emotion, confidence)
        )

# -----------------------------------
# Dominant emotion only
# -----------------------------------

if len(meaningful_emotions) == 1:

    print(

        f"Audience mostly showed "
        f"{meaningful_emotions[0][0]}."

    )

# -----------------------------------
# Mixed emotional response
# -----------------------------------

elif len(meaningful_emotions) >= 2:

    print(

        f"Audience showed "
        f"{meaningful_emotions[0][0]} "

        f"and also expressed "
        f"{meaningful_emotions[1][0]}."

    )
# -----------------------------------
# Display attention weights
# -----------------------------------

#words = text.lower().split() # as senteces now have CLS at the start so add it here in words for display
words = ["[CLS]"] + text.lower().split()

#weights = attention_weights.squeeze().tolist()

#attention = attention_weights.squeeze()

print("\nAttention Matrix:\n")

for head in range(attention_weights.shape[1]):

    print(f"\n========== HEAD {head} ==========\n")

    attention = attention_weights[0][head]

    for i, row in enumerate(attention):

        print(f"\n{words[i]} attends to:\n")

        for j, score in enumerate(row):

            print(
                f"  {words[j]} -> {score.item():.4f}"
            )


