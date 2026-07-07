import torch
import torch.nn as nn
import torch.optim as optim

from ml.audience_intelligence.vocabulary import (
    word_to_index
)
from ml.transformer_decoder.mini_gpt_dataset import (
    input_tensor,
    target_tensor
)
from ml.transformer_decoder.mini_gpt import MiniGPT
# -----------------------------------
# Model configuration
# -----------------------------------

vocab_size = len(word_to_index)

max_seq_length = 12
embedding_dim = 8

# -----------------------------------
# Create model
# -----------------------------------

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    max_seq_length= max_seq_length,
    num_layers=3
)


# -----------------------------------
#  classification loss
# -----------------------------------

criterion = nn.CrossEntropyLoss()
# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# Training loop
# -----------------------------------

#epochs = 500
epochs = 1500 # for new complex dataset
# -----------------------------------
# Flatten target tensor once
# -----------------------------------

target = target_tensor.view(-1)

for epoch in range(epochs):

    # Forward pass
    logits = model(input_tensor)
    #flatten the shape from (batch, sequence,classes) to (number_of_examples, number_of_classes)
    logits = logits.view( -1,vocab_size)
    #target_tensor = target_tensor.view(-1)
    # Compute loss
    loss = criterion(
        logits,
        target
    )

    # Clear old gradients
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress
    if epoch % 50 == 0:

        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.4f}"
        )
        print(f"Logits Shape : {logits.shape}")
        print(f"Target Shape : {target.shape}")
# -----------------------------------
# Save trained model
# -----------------------------------

torch.save(
    model.state_dict(),
    "mini_gpt_model.pt"
)

print("\nModel saved successfully.")


#Evaluation loop

with torch.no_grad():
   

    logits = model(input_tensor)

    predictions = torch.argmax(
       logits,
       dim=-1
    )

    predictions = predictions.view(-1)

    target = target_tensor.view(-1)
    
    print(predictions)
    print(target)

    accuracy = (
        predictions == target
    ).float().mean()
    
    print(accuracy)
