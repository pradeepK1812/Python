import torch
import torch.nn as nn
import torch.optim as optim

from audience_intelligence.vocabulary import (
    word_to_index
)
from audience_intelligence.dataset import (
    X_tensor,
    y_tensor
)
from mini_bert_classifier.mini_bert_classifier import (
    MiniBERTClassifier
)
# -----------------------------------
# Model configuration
# -----------------------------------

vocab_size = len(word_to_index)

max_seq_length = 15
embedding_dim = 8

# -----------------------------------
# Create model
# -----------------------------------

model = MiniBERTClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    max_seq_length=X_tensor.shape[1],
    num_layers=3,
    num_classes=6
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

for epoch in range(epochs):

    # Forward pass
    predictions = model(X_tensor)
    # Compute loss
    loss = criterion(
        predictions,
        y_tensor
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
    #print(predictions.shape)
   # print(y_tensor.shape)
# -----------------------------------
# Save trained model
# -----------------------------------

torch.save(
    model.state_dict(),
    "mini_bert_classifier_model.pt"
)

print("\nModel saved successfully.")

with torch.no_grad():

    logits = model(X_tensor)

    predictions = torch.argmax(
        logits,
        dim=1
    )

    print(predictions)
    print(y_tensor)
    accuracy = (
       predictions == y_tensor
    ).float().mean()

    print(accuracy)
