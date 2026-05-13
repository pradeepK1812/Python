import os

#MODEL_PATH = os.getenv("LINEAR_MODEL_PATH", "model.pt")
#Linear model config
LINEAR_MODEL_PATH = os.getenv(
    "LINEAR_MODEL_PATH",
    "ml/linear_regression/model.pt"
)
API_VERSION = os.getenv("API_VERSION", "v1")

#logistic model config
LOGISTIC_MODEL_PATH = os.getenv(
    "LOGISTIC_MODEL_PATH",
    "ml/logistic_regression/model.pt"
)

HOST = os.getenv("HOST", "0.0.0.0")

PORT = int(os.getenv("PORT", 8000))
