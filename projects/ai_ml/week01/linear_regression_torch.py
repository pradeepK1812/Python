import torch
import time

# -------------------------
# Generate synthetic data
# -------------------------
torch.manual_seed(42)

N = 1_000_000
X = 2 * torch.rand(N, 1)
true_w = 3.5
true_b = 1.2
noise = torch.randn(N, 1) * 0.5

y = true_w * X + true_b + noise

# -------------------------
# Parameters (trainable!)
# -------------------------
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.1
epochs = 50

# -------------------------
# Training loop
# -------------------------
start = time.time()

for epoch in range(epochs):
    # Forward pass (same as NumPy)
    y_pred = w * X + b

    # Loss (same formula)
    loss = torch.mean((y_pred - y) ** 2)

    # Backpropagation (THIS is the magic)
    loss.backward()

    # Gradient descent step
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # Reset gradients (important!)
        w.grad.zero_()
        b.grad.zero_()

end = time.time()

print("Final w:", w.item())
print("Final b:", b.item())
print("Training time:", end - start)

