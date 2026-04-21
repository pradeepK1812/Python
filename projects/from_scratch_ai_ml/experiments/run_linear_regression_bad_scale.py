import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import LinearRegression

np.random.seed(42)

# Feature 1: large scale (1000–5000)
X1 = np.random.rand(100, 1) * 4000 + 1000

# Feature 2: small scale (1–5)
X2 = np.random.rand(100, 1) * 4 + 1

# Combine features
X = np.hstack((X1, X2))

# True relationship
y = 3 * X1.squeeze() + 2 * X2.squeeze() + 5 + np.random.randn(100)

# Train model WITHOUT scaling
model = LinearRegression(learning_rate=0.00000001, n_iterations=1000)
model.fit(X, y)

# Plot cost
plt.plot(model.costs)
plt.title("Cost vs Iterations (No Scaling)")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

print("Weights:", model.weights)
print("Bias:", model.bias)
