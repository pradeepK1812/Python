import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import LinearRegression

# reproducibility
np.random.seed(42)

# dataset
#X = np.random.randn(100, 2) #simple data set
X = np.random.randn(100, 2) * 10 #add more randomeness
true_w = np.array([[3], [2]])
y = X @ true_w + 1

# learning rates to test
learning_rates = [0.0001, 0.01, 0.1, 1]

plt.figure()

for lr in learning_rates:
    model = LinearRegression(learning_rate=lr, n_iterations=1000)
    losses = model.fit(X, y)

    plt.plot(losses, label=f"lr={lr}")

plt.title("Learning Rate Comparison")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("lr_comparison.png")

print("Saved: lr_comparison.png")
