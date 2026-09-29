"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question2.py
Date: 29-09-2026
Description: Fitting MLP models with different number of neurons per layer.
"""

# Imports
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Load in sunspot dataset
data = sm.datasets.sunspots.load_pandas().data

# SUNACTIVITY column for forecasting
X = np.arange(len(data)).reshape(-1, 1)
y = data['SUNACTIVITY'].values

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False,
    random_state=42
)

# Fitting MLP models
mlp_10 = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    solver='adam',
    max_iter=10000,
    random_state=42
).fit(X_train, y_train)

mlp_300 = MLPRegressor(
    hidden_layer_sizes=(300, 300),
    activation='relu',
    solver='adam',
    max_iter=10000,
    random_state=42
).fit(X_train, y_train)

# Evaluate model performance
y_pred_10 = mlp_10.predict(X_test)
y_pred_300 = mlp_300.predict(X_test)

# Print metrics
print('=== MLP with 10 Nodes For Each Hidden Layer ===')
print('MAE:', mean_absolute_error(y_test, y_pred_10))
print('RMSE:', root_mean_squared_error(y_test, y_pred_10))

print('\n=== MLP with 300 Nodes For Each Hidden Layer  ===')
print('MAE:', mean_absolute_error(y_test, y_pred_300))
print('RMSE:', root_mean_squared_error(y_test, y_pred_300))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(X_train, y_train, '-b', label='Training Data')
plt.plot(X_test, y_test, '-r', label='Test Data')
plt.plot(X_test, y_pred_10, '-g', label='MLP 10 Nodes Per Hidden Layer')
plt.plot(X_test, y_pred_300, '-y', label='MLP 300 Nodes Per Hidden Layers')
plt.legend()
plt.title('Training and Test Data with MLP Predictions')
plt.xlabel('Months')
plt.ylabel('Sunspot Activity')
plt.show()