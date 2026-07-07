import torch
import torch.nn as nn
import torch.optim as optim

from ml.audience_intelligence.model import SentimentModel
from ml.audience_intelligence.dataset import X_tensor, y_tensor
from ml.audience_intelligence.vocabulary import word_to_index

# -----------------------------------
# Model configuration
# -----------------------------------

vocab_size = len(word_to_index)

max_seq_length = 15
#embedding_dim = 8
embedding_dim = 64 # to cater new big dataset

# -----------------------------------
# Create model
# -----------------------------------

model = SentimentModel(
    vocab_size,
    max_seq_length,
    embedding_dim
)

# -----------------------------------
#  classification loss
# -----------------------------------

#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = optim.SGD(
    model.parameters(),
    lr=0.001 #changed from 0.1 to 0.001 for complext dataset
)

# -----------------------------------
# Training loop
# -----------------------------------

#epochs = 500
epochs = 1500 # for new complex dataset

for epoch in range(epochs):

    # Forward pass
   # predictions = model(X_tensor)
    predictions, attention_weights = model(X_tensor)
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
    "sentiment_model.pt"
)

print("\nModel saved successfully.")
