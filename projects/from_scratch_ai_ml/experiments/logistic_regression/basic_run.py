import numpy as np
import matplotlib.pyplot as plt
from src.logistic_regression import LogisticRegression

np.random.seed(42)

# -------------------------
# Create dataset
# -------------------------
X = np.random.randn(200, 2)

# simple linear boundary
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

# -------------------------
# Train model
# -------------------------
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
losses = model.fit(X, y)

# -------------------------
# Predictions
# -------------------------
preds = model.predict(X)
accuracy = np.mean(preds == y)

print("Accuracy:", accuracy)

# -------------------------
# Plot loss
# -------------------------
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("logistic_loss.png")

print("Saved: logistic_loss.png")
