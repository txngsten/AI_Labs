"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question5.py
Date: 29-09-2026
Description: Fitting MLP with PyTorch using tanh activation function.
"""

# Imports
import torch
import torch.nn as nn
import numpy as np
import statsmodels.api as sm

from torch.utils.data import DataLoader, TensorDataset

# Load in sunspot dataset
data = sm.datasets.sunspots.load_pandas().data

# SUNACTIVITY column for forecasting
X = np.arange(len(data)).reshape(-1, 1)
y = data['SUNACTIVITY'].values

# Convert into tensors
X_data = torch.from_numpy(X).float()
y_data = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Data split
split = int(0.8 * len(X_data))
X_train, X_test = X_data[:split], X_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

# Scale using train stats only
X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
X_train, X_test = (X_train - X_mean) / X_std, (X_test - X_mean) / X_std
y_train, y_test = (y_train - y_mean) / y_std, (y_test - y_mean) / y_std

# Convert into tensor datasets
train_tensor = TensorDataset(X_train, y_train)
test_tensor = TensorDataset(X_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
test_loader = DataLoader(test_tensor, batch_size=32, shuffle=False)

# Model init
model = nn.Sequential(
    nn.Linear(1, 50),
    nn.Tanh(),
    nn.Linear(50, 25),
    nn.Tanh(),
    nn.Linear(25, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(2000):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

    # Print losses each 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}: train loss {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test) * y_std + y_mean
    actual = y_test * y_std + y_mean
    mse = loss_fn(preds, actual).item()
    mae = torch.mean(torch.abs(preds - actual)).item()

# Print metrics
print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {np.sqrt(mse):.2f}")
print(f"Test MAE: {mae:.2f}")

