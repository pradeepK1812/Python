import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# -------------------------
# 1. Dataset (same as before)
# -------------------------
N = 100
X = torch.randn(N, 2)

#y = (X[:, 0] + X[:, 1] > 0).float().view(-1, 1) #linear 
y = (X[:, 0]**2 + X[:, 1]**2 > 1).float().view(-1, 1) #circular
# -------------------------
# 2. Define model
# -------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 8),   # W1, b1
            nn.ReLU(),         # activation

            nn.Linear(8,8), #second layer
            nn.ReLU(),

            nn.Linear(8, 1),   # W2, b2
            nn.Sigmoid()       # output
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN()

# -------------------------
# 3. Loss + Optimizer
# -------------------------
criterion = nn.BCELoss()              # cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

# -------------------------
# 4. Training loop
# -------------------------
epochs = 1000

for epoch in range(epochs):
    # Forward
    y_pred = model(X)

    # Loss
    loss = criterion(y_pred, y)

    # Backward
    optimizer.zero_grad()  # clear old gradients
    loss.backward()        # autograd computes gradients

    # Update
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

# -------------------------
# 5. Evaluation
# -------------------------
with torch.no_grad():
    preds = (model(X) > 0.5).float()
    accuracy = (preds == y).float().mean()

print("Accuracy:", accuracy.item())
