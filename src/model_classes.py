import torch
from torch import nn, optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import time

from l96_model import RK4, L96_eq1_xdot


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
    def __init__(self, input_size=1, hidden_size=32, num_hidden_layers=3, out_size=1, act_fn: type[nn.Module] = nn.ReLU):
        super().__init__()
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

def train_model(model, loss_fn, loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if len(batch_x.shape) == 1:
            prediction = torch.squeeze(model(torch.unsqueeze(batch_x, 1))) # Add a dimension for the input features
        else:
            prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(loader)

def test_model(model, loss_fn, loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if len(batch_x.shape) == 1:
                prediction = torch.squeeze(model(torch.unsqueeze(batch_x, 1))) # Add a dimension for the input features
            else:
                prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            test_loss += loss.item()
    return test_loss / len(loader)

def fit_model(model, loss_fn, train_loader, test_loader, optimizer, device, epochs=100):
    model = model.to(device)
    train_losses = []
    test_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_model(model, loss_fn, train_loader, optimizer, device)
        test_loss = test_model(model, loss_fn, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:  # Print every 10 epochs and the last epoch
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s")
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    return train_losses, test_losses

class GCM_network:
    def __init__(self, F, network, time_stepping=RK4):
        self.F = F
        self.network = network
        self.time_stepping = time_stepping

    def rhs(self, X, _):
        device = next(self.network.parameters()).device
        if self.network.hidden_layers[0].in_features == 1:
            X_torch = torch.from_numpy(X)
            X_torch = torch.unsqueeze(X_torch, 1).to(device)  # Add a dimension for the input features
        else:
            X_torch = torch.from_numpy(np.expand_dims(X, 0)).to(device)
        return L96_eq1_xdot(X, self.F) + np.squeeze(self.network(X_torch).data.cpu().numpy())
    def __call__(self, X0, dt, nt, param=[0]):
        time, hist, X = (
            dt * np.arange(nt+1),
            np.zeros((nt+1, len(X0))) * np.nan,
            X0.copy()
        )
        hist[0] = X

        for n in range(nt):
            X = self.time_stepping(self.rhs, dt, X, param)
            hist[n+1], time[n+1] = X, dt * (n+1)
        return hist, time
