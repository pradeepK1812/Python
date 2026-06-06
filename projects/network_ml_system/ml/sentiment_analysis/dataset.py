import torch

from sample_data import training_data
from vocabulary import word_to_index

# -----------------------------------
# Convert sentence into token IDs
# -----------------------------------

def tokenize(text):

    words = text.split()

    #token_ids = [] # as now we have added CLS at the start 
    token_ids = [
     word_to_index["[CLS]"]
    ]

    for word in words:

        token_ids.append(
            word_to_index[word]
        )

    return token_ids

# -----------------------------------
# Build dataset tensors
# -----------------------------------

X = []

y = []

for text, label in training_data:

    # Convert sentence to token IDs
    tokens = tokenize(text)

    X.append(tokens)

    y.append(label)


# -----------------------------------
# Find maximum sentence length
# -----------------------------------

max_length = max(len(tokens) for tokens in X)

# -----------------------------------
# Pad all sequences to same length
# -----------------------------------

for tokens in X:

    while len(tokens) < max_length:

        tokens.append(0)

# -----------------------------------
# Convert into tensors
# -----------------------------------

X_tensor = torch.tensor(X)

y_tensor = torch.tensor(y).float().unsqueeze(1)

# -----------------------------------
# Print dataset
# -----------------------------------

print("\nInput Tensor:\n")

#print(X_tensor)
print(X_tensor.tolist())

print("\nLabel Tensor:\n")

print(y_tensor)
