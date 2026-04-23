import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import LinearRegression

# dataset
X = np.random.randn(100, 2)
y = X @ np.array([[3], [2]]) + 1

# scaling configs
datasets = {
    "no_scaling": X,
    "standardized": (X - X.mean(axis=0)) / X.std(axis=0),
}

for name, X_variant in datasets.items():
    model = LinearRegression(learning_rate=0.01)
    losses = model.fit(X_variant, y)

    plt.plot(losses, label=name)

plt.legend()
plt.title("Scaling Comparison")
plt.savefig("scaling_comparison.png")
