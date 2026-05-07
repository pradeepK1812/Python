FastAPI + PyTorch Inference Flow (api.py Explained)
Goal:

This document explains how api.py connects an ML model with an HTTP API using FastAPI and Uvicorn.

Big Picture Architecture

Client (browser / curl)
        ↓ HTTP request
Uvicorn (server)
        ↓
FastAPI (routing + logic)
        ↓
ML Model (PyTorch)
        ↓
Response (JSON)
📄 api.py (Reference Code)
from fastapi import FastAPI
from ml.model import NetworkModel
import torch

app = FastAPI()

model = NetworkModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

@app.post("/predict")
def predict(data: dict):
    x_input = torch.tensor([data["features"]], dtype=torch.float32)

    with torch.no_grad():
        output = model(x_input)

    return {"prediction": output.item()}


🔧 Step-by-Step Explanation
1. Imports
FastAPI → Web framework
NetworkModel → ML model
torch → ML runtime

👉 Bridge between HTTP and ML

2. Create Application
app = FastAPI()
Creates API app
Handles incoming requests
3. Load Model (Startup Phase)
model = NetworkModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()
What happens:
Model architecture created
Weights loaded
Set to inference mode
Why outside function:
Runs once at startup
Avoids reloading per request
4. Define Endpoint
@app.post("/predict")
Registers route /predict
Accepts POST requests
5. Request Handling

Incoming JSON:

{
  "features": [1.0, 2.0]
}

Converted to:

data = {"features": [1.0, 2.0]}
6. Input Processing (Batch Dimension)
x_input = torch.tensor([data["features"]], dtype=torch.float32)
Input	Shape	Meaning
[1,2]	(2,)	❌ invalid
[[1,2]]	(1,2)	✅ correct
7. Disable Gradients
with torch.no_grad():
No training
Faster
Less memory
8. Model Prediction
output = model(x_input)

Internally:

y = XW + b
9. Return Response
return {"prediction": output.item()}
Tensor → scalar
Converted to JSON
🔁 Full Request Lifecycle
Client → HTTP request
        ↓
Uvicorn receives
        ↓
FastAPI routes
        ↓
predict() runs
        ↓
Model inference
        ↓
Response returned
🧠 Key Concepts
FastAPI → defines endpoints
Uvicorn → runs server
Model Serving → ML as API
Batch dimension → required for model input
⚠️ Current Limitations
No validation
No logging
No error handling
🚀 Next Improvements
Pydantic schemas
Logging
Error handling
Docker
🏁 Final Summary

api.py is the serving layer that:

Accepts HTTP requests
Converts input → tensor
Runs ML model
Returns predictions

👉 Connects web world ↔ ML world

💡 Key Takeaway

FastAPI turns your Python function into a web service, and Uvicorn makes it accessible over HTTP.
