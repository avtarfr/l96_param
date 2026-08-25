"""Offline training loop for the NN parameterization."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import model_classes

steps = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]  # Different time steps for datasets
for step in steps:
    # Load the dataset
    data = np.load(f"data/raw/L96_true_data_{step}.npz")
    X_true = data["X_true"]
    xy_true = data["xy_true"]

    # Prepare the dataset for PyTorch
    
    train_size = int(0.8 * step)
    test_size = step - train_size

    X_true_train = X_true[:train_size, :]
    xy_true_train = xy_true[:train_size, :]
    X_true_test = X_true[train_size:, :]
    xy_true_test = xy_true[train_size:, :]

    BATCH_SIZE = step # Use the full dataset as a single batch for training and testing

    dataset_train = Data.TensorDataset(torch.from_numpy(np.reshape(X_true_train, -1)), torch.from_numpy(np.reshape(xy_true_train, -1)))
    dataloader_train = Data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataset_test = Data.TensorDataset(torch.from_numpy(np.reshape(X_true_test, -1)), torch.from_numpy(np.reshape(xy_true_test, -1)))
    dataloader_test = Data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    data_iterator_train = iter(dataloader_train)
    X_iter, subgrid_tend_iter = next(data_iterator_train)
    plt.figure(dpi=150)
    plt.plot(X_iter.numpy(), subgrid_tend_iter.numpy(), 'o', alpha=0.5)
    plt.xlabel("X_true")
    plt.ylabel("Subgrid Tendencies")
    plt.title(f"Training Data (Step: {step})")
    plt.savefig(f"results/training_data_step_{step}.png")

    fcnn_model_relu = model_classes.FlexibleFCNN(input_size=1, hidden_size=32, num_hidden_layers=3, out_size=1, act_fn=nn.ReLU)
    fcnn_model_tanh = model_classes.FlexibleFCNN(input_size=1, hidden_size=32, num_hidden_layers=3, out_size=1, act_fn=nn.Tanh)

    loss_fn = nn.MSELoss()
    n_epochs = 500
    lr = 0.003
    optimizer_relu = optim.Adam(fcnn_model_relu.parameters(), lr=lr)
    optimizer_tanh = optim.Adam(fcnn_model_tanh.parameters(), lr=lr)

    train_losses_relu, test_losses_relu = model_classes.fit_model(fcnn_model_relu, loss_fn, dataloader_train, dataloader_test, optimizer_relu, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), epochs=n_epochs)
    train_losses_tanh, test_losses_tanh = model_classes.fit_model(fcnn_model_tanh, loss_fn, dataloader_train, dataloader_test, optimizer_tanh, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), epochs=n_epochs)

    # Plotting the training and testing losses for both models, rather than displaying them in the console, we save the plots to files for later review. This is especially useful when running multiple experiments or when the training process is long.
    plt.figure(dpi=150)
    plt.plot(train_losses_relu, label="Train Loss (ReLU)", color='blue', linestyle='--')
    plt.plot(test_losses_relu, label="Test Loss (ReLU)", color='orange', linestyle='--')
    plt.plot(train_losses_tanh, label="Train Loss (Tanh)", color='green')
    plt.plot(test_losses_tanh, label="Test Loss (Tanh)", color='red')
    plt.legend()
    plt.title(f"Training and Testing Losses (Step: {step})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"results/losses_step_{step}.png")
    predictions_relu = fcnn_model_relu(torch.unsqueeze(torch.from_numpy(np.reshape(X_true_test[:, 1], -1)), 1).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))).cpu().detach().numpy()
    predictions_tanh = fcnn_model_tanh(torch.unsqueeze(torch.from_numpy(np.reshape(X_true_test[:, 1], -1)), 1).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))).cpu().detach().numpy()

    plt.figure(dpi=150)
    plt.plot(predictions_relu, label="Predictions (ReLU)", color='blue')
    plt.plot(predictions_tanh, label="Predictions (Tanh)", color='green')
    plt.plot(xy_true_test[:1000, 1], label="True Subgrid Tendencies", color='red', linestyle='--')
    plt.xlim(0, 1000)
    plt.legend()
    plt.title(f"Model Predictions vs True Values (Step: {step})")
    plt.xlabel("Sample Index")
    plt.ylabel("Subgrid Tendencies")
    plt.savefig(f"results/predictions_step_{step}.png")

    # Next, we save the trained models for future use. This allows us to load the models later without retraining, which is especially useful for deployment or further analysis.
    torch.save(fcnn_model_relu.state_dict(), f"models/fcnn_model_relu_step_{step}.pth")
    torch.save(fcnn_model_tanh.state_dict(), f"models/fcnn_model_tanh_step_{step}.pth")