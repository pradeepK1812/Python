from fastapi import FastAPI
from ml.model import NetworkModel
import torch

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
##############################################################################################
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Network ML API</title>
            <style>
                body {
                    font-family: Arial;
                    padding: 40px;
                    background-color: #f4f4f4;
                }

                .container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    max-width: 600px;
                }

                a {
                    color: blue;
                    text-decoration: none;
                }
            </style>
        </head>

        <body>
            <div class="container">
                <h1>🚀 Network ML API</h1>
                <p>API server is running successfully.</p>

                <p>
                    Open
                    <a href="/docs">/docs</a>
                    for Swagger UI documentation.
                </p>
            </div>
        </body>
    </html>
    """
####################################################################################

# Load model once at startup
model = NetworkModel()
model.load_state_dict(torch.load("ml/model.pt"))
model.eval()
logger.info("Model loaded successfully")

@app.on_event("startup")
def startup_event():
    logger.info("API service started")
#@app.post("/predict")

#from app.schemas import PredictionRequest
from app.schemas import PredictionRequest, PredictionResponse
from fastapi import HTTPException


@app.get("/")
@app.get("/health")
def health():
    return {
        "service": "Network ML API",
        "version": "1.0",
        "status": "running"
    }
@app.post("/predict", response_model=PredictionResponse)
#@app.post("/predict")
def predict(data: PredictionRequest):
#def predict(data: dict):
     try:
        logger.info(f"Received input: {data.features}")

       # x_input = torch.tensor([data.features], dtype=torch.float32)
        x_input = torch.tensor(data.features, dtype=torch.float32)

        with torch.no_grad():
            output = model(x_input)
        prediction = output.squeeze().tolist()
        return PredictionResponse(prediction=prediction)
       # return PredictionResponse(prediction=output.item())

     except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
   
