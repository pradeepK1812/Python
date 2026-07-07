# -----------------------------------
# Create GPT training pairs
#
# Input :
#  [CLS] I love
#
# Target:
#  I love transformers
# -----------------------------------
import torch

from ml.audience_intelligence.sample_data import (
    training_data
)

from ml.audience_intelligence.vocabulary import (
    word_to_index
)

#tokenizer 

def tokenize(text):

    words = text.lower().split()

    token_ids = [
        word_to_index["[CLS]"]
    ]

    for word in words:

        if word in word_to_index:

            token_ids.append(
                word_to_index[word]
            )

    return token_ids

#dataset creation 
input_sequences = []

target_sequences = []

for text, label in training_data:
    tokens = tokenize(text)
    input_sequences.append(
     tokens[:-1]
    )

    target_sequences.append(
      tokens[1:]
    )

# -----------------------------------
# Find maximum sequence length
# -----------------------------------

max_length = max(

    len(sequence)

    for sequence in input_sequences
)

for sequence in input_sequences:

    while len(sequence) < max_length:

        sequence.append(0)

for sequence in target_sequences:

    while len(sequence) < max_length:

        sequence.append(0)

input_tensor = torch.tensor(
    input_sequences
)

target_tensor = torch.tensor(
    target_sequences
)
"""
print("\nInput Tensor:\n")
print(input_tensor.tolist())

print("\nTarget Tensor:\n")
print(target_tensor.tolist())
"""
print("\nInput Shape:")
print(input_tensor.shape)

print("\nTarget Shape:")
print(target_tensor.shape)

print("\nFirst Training Example")

print("Input :", input_tensor[0].tolist())
print("Target:", target_tensor[0].tolist())
