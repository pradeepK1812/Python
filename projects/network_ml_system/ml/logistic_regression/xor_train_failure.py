import torch
import torch.nn as nn
import torch.optim as optim

from model import LogisticRegressionModel

# -------------------------
# XOR dataset
# -------------------------

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# -------------------------
# Model
# -------------------------

model = LogisticRegressionModel()

criterion = nn.BCELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)

# -------------------------
# Training
# -------------------------

epochs = 1000

for epoch in range(epochs):

    # Forward pass: model predicts probabilities
    predictions = model(X)

    # Compute prediction error
    loss = criterion(predictions, y)

    # Clear old gradients from previous step
    optimizer.zero_grad()

    # Compute new gradients using backpropagation
    loss.backward()

    # Update weights and bias using gradients
    optimizer.step()

    # Print training progress periodically
    if epoch % 100 == 0:

        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.4f}"
        )
# -------------------------
# Final predictions
# -------------------------

with torch.no_grad():

    predictions = model(X)

print("\nPredictions:")

print(predictions)
