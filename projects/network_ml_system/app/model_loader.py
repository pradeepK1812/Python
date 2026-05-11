from ml.model import NetworkModel
import torch

model = NetworkModel()

model.load_state_dict(torch.load("ml/model.pt"))

model.eval()
