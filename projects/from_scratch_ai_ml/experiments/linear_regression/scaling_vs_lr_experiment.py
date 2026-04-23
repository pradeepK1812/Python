import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import LinearRegression

np.random.seed(42)

# dataset (large scale)
X = np.random.randn(100, 2) * 10
true_w = np.array([[3], [2]])
y = X @ true_w + 1

# normalized version
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

learning_rates = [0.0001, 0.001, 0.01,]

plt.figure(figsize=(10, 5))

# -------- Unscaled --------
plt.subplot(1, 2, 1)
for lr in learning_rates:
    model = LinearRegression(learning_rate=lr, n_iterations=500)
    losses = model.fit(X, y)
    plt.plot(losses, label=f"lr={lr}")

plt.title("Unscaled Data")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

# -------- Normalized --------
plt.subplot(1, 2, 2)
for lr in learning_rates:
    model = LinearRegression(learning_rate=lr, n_iterations=500)
    losses = model.fit(X_norm, y)
    plt.plot(losses, label=f"lr={lr}")

plt.title("Normalized Data")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("scaling_vs_lr.png")

print("Saved: scaling_vs_lr.png")
