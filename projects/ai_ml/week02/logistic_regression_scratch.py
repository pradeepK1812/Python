import numpy as np

# -----------------------
# 1. Generate binary data
# -----------------------
np.random.seed(0)
N = 1000

X = np.random.randn(N, 1)

true_w = 2.0
true_b = -0.5

z = true_w * X + true_b
y = (z > 0).astype(float)   # labels: 0 or 1

# -----------------------
# 2. Normalize (important!)
# -----------------------
X = (X - X.mean()) / X.std()

# -----------------------
# 3. Model parameters
# -----------------------
w = np.random.randn()
b = 0.0

lr = 0.1
epochs = 300

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------
# 4. Training loop
# -----------------------
for epoch in range(epochs):
    z = w * X + b
    p = sigmoid(z)

    # Cross-entropy gradient simplifies to (p - y)
    dw = np.mean((p - y) * X)
    db = np.mean(p - y)

    w -= lr * dw
    b -= lr * db

    if epoch % 50 == 0:
        loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
        print(f"Epoch {epoch:3d} | Loss {loss:.4f}")

# -----------------------
# 5. Evaluate
# -----------------------
preds = (sigmoid(w * X + b) > 0.5).astype(float)
accuracy = np.mean(preds == y)

print("\nFinal parameters:")
print("w =", round(w,3), "b =", round(b,3))
print("Accuracy:", round(accuracy,3))

