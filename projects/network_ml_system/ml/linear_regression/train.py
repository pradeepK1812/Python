from model import NetworkModel
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Generate synthetic data
# -------------------------
torch.manual_seed(42)

X = torch.randn(100, 2)
y = X @ torch.tensor([[3.0], [2.0]]) + 1.0

# -------------------------
# Model, Loss, Optimizer
# -------------------------
model = NetworkModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -------------------------
# Training loop
# -------------------------
epochs = 500

for epoch in range(epochs):
    model.train()

    # Forward pass
    preds = model(X)
    loss = loss_fn(preds, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
# train.py
torch.save(model.state_dict(), "model.pt")
# -------------------------
# Check learned weights
# -------------------------
for name, param in model.named_parameters():
    print(name, param.data)
