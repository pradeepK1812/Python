from fastapi import FastAPI
from ml.model import NetworkModel
import torch

app = FastAPI()

# Load model once at startup
model = NetworkModel()
model.load_state_dict(torch.load("ml/model.pt"))
model.eval()

@app.post("/predict")
def predict(data: dict):
    x_input = torch.tensor([data["features"]], dtype=torch.float32)

    with torch.no_grad():
        output = model(x_input)

    return {"prediction": output.item()}
