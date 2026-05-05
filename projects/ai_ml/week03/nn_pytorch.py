import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
            nn.Linear(2, 32),   # W1, b1
            nn.ReLU(),         # activation

            nn.Linear(32,32), #second layer
            nn.ReLU(),

            nn.Linear(32, 1),   # W2, b2
            nn.Sigmoid()       # output
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN()

# -------------------------
# 3. Loss + Optimizer
# -------------------------
criterion = nn.BCELoss()              # cross entropy
#optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

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


import numpy as np

# create grid
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)

with torch.no_grad():
    Z = model(grid_tensor)
    Z = Z.reshape(xx.shape).numpy()


plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="coolwarm", edgecolors='k')
plt.title("PyTorch Decision Boundary")
plt.savefig("pytorch_decision_boundary.png")
plt.show()
