from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


app = FastAPI()

app.include_router(health_router)
app.include_router(predict_router)
