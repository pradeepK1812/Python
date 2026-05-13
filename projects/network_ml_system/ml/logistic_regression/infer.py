import torch

from model import LogisticRegressionModel

# -------------------------
# Load model
# -------------------------

model = LogisticRegressionModel()

model.load_state_dict(torch.load("model.pt"))

model.eval()

# -------------------------
# Test inputs
# -------------------------

X_test = torch.tensor([
    [5.0, 5.0],
    [-5.0, -5.0],
    [1.0, -0.5],
    [-1.0, 0.2]
])

# -------------------------
# Inference
# -------------------------

with torch.no_grad():

    probabilities = model(X_test)

# -------------------------
# Convert probability to class
# -------------------------

predicted_classes = (probabilities >= 0.5).float()

# -------------------------
# Print results
# -------------------------

print("Probabilities:")
print(probabilities)

print("\nPredicted Classes:")
print(predicted_classes)
