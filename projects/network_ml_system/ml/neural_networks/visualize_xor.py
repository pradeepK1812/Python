import torch
import matplotlib.pyplot as plt

from model import XORNeuralNetwork

# -----------------------------------
# XOR dataset
# -----------------------------------

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    0,
    1,
    1,
    0
])

# -----------------------------------
# Load trained model
# -----------------------------------

model = XORNeuralNetwork()

model.load_state_dict(
    torch.load("xor_model.pt")
)

model.eval()

# -----------------------------------
# Create grid for visualization
# -----------------------------------

x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5

xx, yy = torch.meshgrid(
    torch.linspace(x_min, x_max, 200),
    torch.linspace(y_min, y_max, 200),
    indexing='ij'
)

# -----------------------------------
# Flatten grid into input points
# -----------------------------------

grid = torch.cat([
    xx.reshape(-1, 1),
    yy.reshape(-1, 1)
], dim=1)

# -----------------------------------
# Predict probabilities over grid
# -----------------------------------

with torch.no_grad():

    probs = model(grid)

# -----------------------------------
# Reshape probabilities into 2D map
# -----------------------------------

Z = probs.reshape(xx.shape)

# -----------------------------------
# Plot decision boundary
# -----------------------------------

#for proper color organization

plt.contourf(
    xx.numpy(),
    yy.numpy(),
    Z.numpy(),
    levels=[0, 0.5, 1],
    alpha=0.3
)
# -----------------------------------
# Plot XOR points
# -----------------------------------

for i in range(len(X)):

    if y[i].item() == 0: # convert tensor to python scaler for correct plotting

        plt.scatter(
            X[i][0],
            X[i][1],
            label="Class 0" if i == 0 else ""
        )

    else:

        plt.scatter(
            X[i][0],
            X[i][1],
            label="Class 1" if i == 1 else ""
        )

# -----------------------------------
# Labels
# -----------------------------------

plt.title("XOR Neural Network Decision Boundary")

plt.xlabel("x1")

plt.ylabel("x2")

plt.legend()

# -----------------------------------
# Save figure
# -----------------------------------

plt.savefig(
    "xor_decision_boundary.png",
    dpi=300,
    bbox_inches="tight"
)
