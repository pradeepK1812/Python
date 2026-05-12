from ml.model import NetworkModel
import torch
from app.config import MODEL_PATH

model = NetworkModel()

#model.load_state_dict(torch.load("ml/model.pt"))

model.load_state_dict(torch.load(MODEL_PATH))

model.eval()
