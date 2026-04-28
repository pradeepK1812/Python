import numpy as np
import matplotlib.pyplot as plt
from src.logistic_regression import LogisticRegression

np.random.seed(42)
#Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)

    Z = probs.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, levels=50)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k')
    plt.title("Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("logistic_boundary.png")


# -------------------------
# Create dataset
# -------------------------
X = np.random.randn(200, 2)

# simple linear boundary
# y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1) #linear dataset 
y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int).reshape(-1, 1) # circular 
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
plot_decision_boundary(X, y, model)
print("Saved: logistic_boundary.png")
