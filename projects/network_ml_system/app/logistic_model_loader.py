import torch

from app.config import LOGISTIC_MODEL_PATH

from ml.logistic_regression.model import LogisticRegressionModel

# -------------------------
# Load model
# -------------------------

model = LogisticRegressionModel()

model.load_state_dict(
    torch.load(LOGISTIC_MODEL_PATH)
)

model.eval()
