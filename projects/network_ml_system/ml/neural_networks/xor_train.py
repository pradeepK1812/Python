import torch
import torch.nn as nn
import torch.optim as optim

from model import XORNeuralNetwork

torch.manual_seed(42)
# -----------------------------------
# XOR dataset
# -----------------------------------

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

# -----------------------------------
# Create neural network
# -----------------------------------

model = XORNeuralNetwork()

# -----------------------------------
# Binary classification loss
# -----------------------------------

criterion = nn.BCELoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)

# -----------------------------------
# Training loop
# -----------------------------------

epochs = 5000

for epoch in range(epochs):

    # Forward pass
    predictions = model(X)

    # Compute error
    loss = criterion(predictions, y)

    # Clear previous gradients
    optimizer.zero_grad()

    # Compute new gradients
    loss.backward()

    # Update weights
    optimizer.step()

    # Print progress
    if epoch % 500 == 0:

        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.4f}"
        )
    torch.save(
    model.state_dict(),
    "xor_model.pt"
    ) 
# -----------------------------------
# Final predictions
# -----------------------------------

with torch.no_grad():

    predictions = model(X)

print("\nFinal Predictions:")

print(predictions)
