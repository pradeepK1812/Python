import time
import uuid
import logging

import torch

from fastapi import APIRouter, HTTPException

from app.schemas import PredictionRequest
from app.logistic_model_loader import model

from app.config import API_VERSION

router = APIRouter()

logger = logging.getLogger(__name__)

# -------------------------
# Logistic prediction route
# -------------------------

@router.post("/predict-logistic")

def predict(data: PredictionRequest):

    request_id = str(uuid.uuid4())

    start_time = time.time()

    try:

        logger.info(
            f"[{request_id}] "
            f"Version={API_VERSION} "
            f"Input={data.features}"
        )

        # Convert input to tensor
        x_input = torch.tensor(
            data.features,
            dtype=torch.float32
        )

        # Inference
        with torch.no_grad():

            probabilities = model(x_input)

        # Convert probabilities to classes
        predicted_classes = (
            probabilities >= 0.5
        ).float()

        probability_output = probabilities.squeeze().tolist()

        class_output = predicted_classes.squeeze().tolist()

        elapsed_time = time.time() - start_time

        logger.info(
            f"[{request_id}] "
            f"Version={API_VERSION} "
            f"Output={class_output}"
        )

        logger.info(
            f"[{request_id}] "
            f"Version={API_VERSION} "
            f"Completed in {elapsed_time:.4f}s"
        )

        return {
            "probabilities": probability_output,
            "predicted_classes": class_output
        }

    except Exception as e:

        logger.error(
            f"[{request_id}] "
            f"Prediction failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
