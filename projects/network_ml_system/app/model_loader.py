from ml.linear_regression.model import NetworkModel
import torch
from app.config import LINEAR_MODEL_PATH

model = NetworkModel()

#model.load_state_dict(torch.load("ml/model.pt"))

model.load_state_dict(torch.load(LINEAR_MODEL_PATH))

model.eval()
