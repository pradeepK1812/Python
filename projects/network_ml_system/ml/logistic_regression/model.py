import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    
    def __init__(self):

        #Initialize the parent class nn.module
        super().__init__()

        self.linear = nn.Linear(2, 1)

    def forward(self, x):

        logits = self.linear(x)

        probabilities = torch.sigmoid(logits)

        return probabilities
