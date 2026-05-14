import torch
import torch.nn as nn


class XORNeuralNetwork(nn.Module):

    def __init__(self):

        # Initialize parent nn.Module
        super().__init__()

        # -----------------------------------
        # First layer:
        # Takes 2 input features
        # Produces 4 hidden features
        # -----------------------------------
        self.hidden = nn.Linear(2, 4)
        

        #  -----------------------------------
        # ReLU activation introduces
        # nonlinearity into the network
        # -----------------------------------
        
        # Replaced ReLU with Tanh because
        # Tanh outputs values between -1 and 1,
        # which works better for this small XOR problem
        self.relu = nn.Tanh()
       
        # -----------------------------------
        # Second layer:
        # Takes 4 hidden features
        # Produces 1 output value
        # -----------------------------------
        self.output = nn.Linear(4, 1)

       
        # -----------------------------------
        # Sigmoid converts final output
        # into probability between 0 and 1
        # -----------------------------------
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # -----------------------------------
        # Pass input through hidden layer
        # -----------------------------------
        x = self.hidden(x)

        # -----------------------------------
        # Apply nonlinear activation
        # This is the key difference from
        # logistic regression
        # -----------------------------------
        x = self.relu(x)

        # -----------------------------------
        # Pass activated features to output
        # layer for final decision score
        # -----------------------------------
        x = self.output(x)

        # -----------------------------------
        # Convert score into probability
        # -----------------------------------
        x = self.sigmoid(x)

        return x
