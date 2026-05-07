from fastapi import FastAPI
from ml.model import NetworkModel
import torch

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model once at startup
model = NetworkModel()
model.load_state_dict(torch.load("ml/model.pt"))
model.eval()

#@app.post("/predict")

#from app.schemas import PredictionRequest
from app.schemas import PredictionRequest, PredictionResponse
from fastapi import HTTPException

@app.post("/predict", response_model=PredictionResponse)
#@app.post("/predict")
def predict(data: PredictionRequest):
#def predict(data: dict):
     try:
        logger.info(f"Received input: {data.features}")

        x_input = torch.tensor([data.features], dtype=torch.float32)

        with torch.no_grad():
            output = model(x_input)

        return PredictionResponse(prediction=output.item())

     except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
   
