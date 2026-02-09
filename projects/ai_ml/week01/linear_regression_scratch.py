import numpy as np

# -------------------------
# Generate synthetic data
# -------------------------
np.random.seed(42)

N = 1000
X = 2 * np.random.rand(N, 1)
true_w = 3.5
true_b = 1.2
noise = np.random.randn(N, 1) * 0.5

y = true_w * X + true_b + noise

# -------------------------
# Initialize parameters
# -------------------------
w = np.random.randn()
b = 0.0

learning_rate = 0.1
epochs = 50

# -------------------------
# Training loop
# -------------------------
for epoch in range(epochs):
    y_pred = w * X + b

    # Mean Squared Error
    loss = np.mean((y_pred - y) ** 2)

    # Gradients
    dw = 2 * np.mean((y_pred - y) * X)
    db = 2 * np.mean(y_pred - y)

    # Update
    w -= learning_rate * dw
    b -= learning_rate * db

    print(f"Epoch {epoch:02d} | Loss={loss:.4f} | w={w:.3f} | b={b:.3f}")

print("\nFinal parameters:")
print(f"w={w:.3f}, b={b:.3f}")

