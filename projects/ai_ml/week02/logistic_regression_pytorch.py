import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Generate data
# -----------------------
torch.manual_seed(0)
N = 1000

X = torch.randn(N, 1)
true_w = 2.0
true_b = -0.5

z = true_w * X + true_b
y = (z > 0).float()

# -----------------------
# Normalize
# -----------------------
X = (X - X.mean()) / X.std()

# -----------------------
# Model
# -----------------------
model = nn.Linear(1, 1)

criterion = nn.BCEWithLogitsLoss()  
# (sigmoid + cross entropy combined for stability)

optimizer = optim.SGD(model.parameters(), lr=0.1)

# -----------------------
# Training loop
# -----------------------
epochs = 300

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(X)
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss {loss.item():.4f}")

# -----------------------
# Evaluate
# -----------------------
with torch.no_grad():
    probs = torch.sigmoid(model(X))
    preds = (probs > 0.5).float()
    accuracy = (preds == y).float().mean()

print("\nLearned w:", model.weight.item())
print("Learned b:", model.bias.item())
print("Accuracy:", accuracy.item())

