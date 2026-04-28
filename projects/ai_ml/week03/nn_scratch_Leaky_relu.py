import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#Plotting the nurons as per the activation

def plot_hidden_neurons(X, forward_full, hidden_dim):
    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get hidden layer activations
    z1, a1, _, _ = forward_full(grid)

    for i in range(hidden_dim):
        neuron_output = a1[:, i].reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, neuron_output, levels=50)
        plt.scatter(X[:, 0], X[:, 1], edgecolors='k')
        plt.title(f"Hidden Neuron {i}")
        plt.xlabel("x1")
        plt.ylabel("x2")

        filename = f"hidden__leakyrelu_neuron_{i}.png"
        plt.savefig(filename)
        plt.close()

        print("Saved:", filename)
# -------------------------
# Decision boundary plot
# -------------------------

def plot_decision_boundary(X, y, forward_fn):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    #probs = forward_fn(grid)
    #as forward _full returns a touple so we need to extract only probability here
    _, _, _, probs = forward_fn(grid)
    Z = probs.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, levels=50)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k')
    plt.title("Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("decision_boundary_lkrelu.png")
    plt.close()


# -------------------------
# 1. Create dummy dataset
# -------------------------
N = 100
X = np.random.randn(N, 2)

# simple rule: class 1 if x1 + x2 > 0
#y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int).reshape(-1, 1)
# -------------------------
# 2. Initialize parameters
# -------------------------
input_dim = 2
hidden_dim = 4
output_dim = 1

W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))

W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

# -------------------------
# 3. Activation functions
# -------------------------
#def relu(x):
 #   return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------
# 4. Forward pass
# -------------------------
def forward_full(X):
    z1 = X @ W1 + b1
    #a1 = relu(z1)
    a1 = leaky_relu(z1)
    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)
    return z1, a1, z2, y_pred
#-----------------------------
#-----loss computation------#
#--------------------------------
def compute_loss(y, y_pred):
    eps = 1e-8  # avoid log(0)
    loss = -np.mean(
        y * np.log(y_pred + eps) +
        (1 - y) * np.log(1 - y_pred + eps)
    )
    return loss

#------------------------------
# RELU Gred1 
#---------------------------------
#def relu_grad(x):
 #   return (x > 0).astype(float)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)
# -------------------------
# 5. Run once
# -------------------------
#y_pred = forward(X)
epochs = 1000
learning_rate = 0.1
losses = []

for epoch in range(epochs):
    # Forward
    z1, a1, z2, y_pred = forward_full(X)
    # Loss
    loss = compute_loss(y, y_pred)
    losses.append(loss)
    # Backprop
    dz2 = y_pred - y

    dW2 = a1.T @ dz2 / N
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
   # dz1 = da1 * relu_grad(z1)
    dz1 = da1 * leaky_relu_grad(z1)

    dW1 = X.T @ dz1 / N
    db1 = np.mean(dz1, axis=0, keepdims=True)
    
    print("||dW1||:", np.linalg.norm(dW1), "||dW2||:", np.linalg.norm(dW2)) 
    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch:04d} | Loss {loss:.4f}")

print("Sample predictions:\n", y_pred[:5])
preds = (y_pred > 0.5).astype(int)
accuracy = np.mean(preds == y)
print("Accuracy:", accuracy)

# -------------------------
# Graphical representation
# -------------------------

plot_decision_boundary(X, y, forward_full)
print("Saved: decision_boundary.png")
plot_hidden_neurons(X, forward_full, hidden_dim)
