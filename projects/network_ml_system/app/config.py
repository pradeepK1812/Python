import os

MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")

API_VERSION = os.getenv("API_VERSION", "v1")

HOST = os.getenv("HOST", "0.0.0.0")

PORT = int(os.getenv("PORT", 8000))
