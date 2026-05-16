import torch
import torch.nn as nn
import torch.optim as optim

#inport logisticregression model from model.py
from model import LogisticRegressionModel


# Reproducibility
torch.manual_seed(42)

# -------------------------
# Synthetic classification data
# -------------------------

N = 100

# Features
X = torch.randn(N, 2)

# Labels
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

#print("Features:")
#print(X[:5])

#print("\nLabels:")
#print(y[:5])

# -------------------------
# Model
# -------------------------

model = LogisticRegressionModel()

# -------------------------
# Loss and optimizer
# -------------------------

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)

# -------------------------
# Training loop
# -------------------------

epochs = 400

for epoch in range(epochs):

    # Forward pass
    predictions = model(X)
    #compute loss wrt to prediction and actual different 
    loss = criterion(predictions, y)

    # Backpropagation

    #clear the old gradients from previous iterations
    optimizer.zero_grad()
    #compute gradients using  backpropagation
    loss.backward()
    # Gradient descent :Update weights and bias using:
    # parameter = parameter - (learning_rate * gradient)
    # Example:
    # w = w - lr * (dL/dw)
    optimizer.step()

    if epoch % 10 == 0:

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------
# Save model
# -------------------------

torch.save(
    model.state_dict(),
    "model.pt"
)

print("\nModel saved successfully.")
