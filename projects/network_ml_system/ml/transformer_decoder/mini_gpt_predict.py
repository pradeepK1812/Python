import torch

from ml.audience_intelligence.vocabulary import word_to_index
from ml.transformer_decoder.mini_gpt  import MiniGPT

# -----------------------------------
# Reverse vocabulary
# -----------------------------------

index_to_word = {

    index: word

    for word, index in word_to_index.items()

}

# -----------------------------------
# Model configuration
# -----------------------------------

vocab_size = len(word_to_index)

embedding_dim = 8
max_seq_length = 12
# -----------------------------------
# Create model
# -----------------------------------

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_dim=8,
    max_seq_length=max_seq_length,
    num_layers=3
)
# -----------------------------------
# Load trained weights
# -----------------------------------

model.load_state_dict(
    torch.load(
        "mini_gpt_model.pt"
    )
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

text = "this tutorial is"
print(f"\nInput sentence is :{text} \n")
# -----------------------------------
# Convert to token IDs
# -----------------------------------

tokens = tokenize(text)

print("\nToken IDs:\n")

print(tokens)

#max tokens to generate for each new prediction
max_new_tokens = 5
# -----------------------------------
# Convert to tensor (moved inside loop)
# -----------------------------------

#input_tensor = torch.tensor( [tokens])

# -----------------------------------
# Run inference
# -----------------------------------

with torch.no_grad():

    for _ in range(max_new_tokens):


        input_tensor = torch.tensor( [tokens])
        logits = model(input_tensor)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(
             next_token_logits,
             dim=-1
        )
        # predicted_word = index_to_word[next_token.item()]
        next_id = next_token.item()
        tokens.append(next_id)
        print(f"Next word: {index_to_word[next_id]}")

