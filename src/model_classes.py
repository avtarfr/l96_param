import torch
from torch import nn, optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

"""
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 1)

        self.activation = nn.ReLU()  # Use ReLU activation for non-smoothness

        nn.init.zeros_(self.linear4.weight)  # Initialize the last layer weights to zero
        nn.init.zeros_(self.linear4.bias)    # Initialize the last layer bias to zero

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x # non-smooth activating functions like ReLU can introduce non-smoothness in the output, which might not be desirable for certain applications. To address this, we can modify the FCNN architecture to use smooth activation functions, such as Tanh or Sigmoid, which can help maintain smoothness in the output.

class FCNN_smooth(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 1)

        self.activation = nn.Tanh()  # Use Tanh activation for smoothness

        nn.init.zeros_(self.linear4.weight)  # Initialize the last layer weights to zero
        nn.init.zeros_(self.linear4.bias)    # Initialize the last layer bias to zero
        # The setting to zero ensures that the network starts with a 0 subgrid forcing, and the Tanh activation helps in maintaining smoothness.

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x
"""
# generalizing
class FlexibleFCNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_hidden_layers=3, out_size=1, act_fn=nn.Tanh):
        layers = []
        # Input Layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(act_fn())

        # Hidden Layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act_fn())

        self.output_layer = nn.Linear(hidden_size, out_size)
        nn.init.zeros_(self.output_layer.weight)  # Initialize the last layer weights to zero
        nn.init.zeros_(self.output_layer.bias)    # Initialize the last layer bias to zero

        self.hidden_layers = nn.Sequential(*layers) # This line creates a sequential container of the hidden layers and activation functions. This means that when you pass an input through self.hidden_layers, it will sequentially go through all the layers and activation functions defined in the layers list.

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

