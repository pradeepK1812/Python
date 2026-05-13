import torch
import matplotlib.pyplot as plt

from model import LogisticRegressionModel

# -------------------------
# Generate dataset
# -------------------------

torch.manual_seed(42)

N = 100

X = torch.randn(N, 2)

y = (X[:, 0] + X[:, 1] > 0).float()

# -------------------------
# Load trained model
# -------------------------

model = LogisticRegressionModel()

model.load_state_dict(torch.load("model.pt"))

model.eval()

# -------------------------
# Extract weights and bias
# -------------------------

weights = model.linear.weight.detach()[0]

bias = model.linear.bias.detach()

w1 = weights[0].item()
w2 = weights[1].item()

b = bias.item()

# -------------------------
# Decision boundary
# -------------------------

x_values = torch.linspace(-3, 3, 100)

y_values = -(w1 * x_values + b) / w2

# -------------------------
# Plot points
# -------------------------

class0 = y == 0
class1 = y == 1

plt.scatter(
    X[class0][:, 0],
    X[class0][:, 1],
    label="Class 0"
)

plt.scatter(
    X[class1][:, 0],
    X[class1][:, 1],
    label="Class 1"
)

# -------------------------
# Plot boundary
# -------------------------

plt.plot(
    x_values,
    y_values,
    label="Decision Boundary"
)

plt.xlabel("x1")

plt.ylabel("x2")

plt.legend()

plt.title("Logistic Regression Decision Boundary")

plt.savefig("decision_boundary.png")
