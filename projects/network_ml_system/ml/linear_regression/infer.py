from model import NetworkModel
import torch

# -------------------------
# Load model
# -------------------------
model = NetworkModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()   # VERY IMPORTANT

# -------------------------
# Prediction function
# -------------------------
def predict(x_input):
    x = torch.tensor(x_input, dtype=torch.float32)

    with torch.no_grad():   # no gradients needed
        output = model(x)

    return output.numpy()

# -------------------------
# Test inference
# -------------------------
if __name__ == "__main__":
    test_input = [[1.0, 2.0]]
    pred = predict(test_input)
    print("Prediction:", pred)
