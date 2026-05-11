from fastapi import APIRouter, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import model

import torch
import logging
import uuid
import time

router = APIRouter()

logger = logging.getLogger(__name__)
#logger = logging.getLogger("uvicorn")
#logger.setLevel(logging.INFO)

@router.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    request_id = str(uuid.uuid4())
    start = time.time()

    try:
        print("PREDICT FUNCTION HIT")
        logger.info(f"[{request_id}] Input: {data.features}")

        x_input = torch.tensor(data.features, dtype=torch.float32)

        with torch.no_grad():
            output = model(x_input)

        prediction = output.squeeze().tolist()

        duration = time.time() - start

        logger.info(f"[{request_id}] Completed in {duration:.4f}s")

        return PredictionResponse(prediction=prediction)

    except Exception as e:

        logger.error(f"[{request_id}] Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
