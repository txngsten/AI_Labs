"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: questions1.py
Date: 25-09-2026
Description:
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
mlp_1_hidden = MLPRegressor(
    hidden_layer_sizes=(50,),
    activation='relu',
    solver='adam',
    max_iter=10000
).fit(X_train, y_train)

mlp_5_hidden = MLPRegressor(
    hidden_layer_sizes=(50, 50, 50, 50, 50),
    activation='relu',
    solver='adam',
    max_iter=10000
).fit(X_train, y_train)

# Evaluate model performance
y_pred_1 = mlp_1_hidden.predict(X_test)
y_pred_5 = mlp_5_hidden.predict(X_test)

# Print metrics
print('=== MLP with 1 Hidden Layer ===')
print('MAE:', mean_absolute_error(y_test, y_pred_1))
print('RMSE:', root_mean_squared_error(y_test, y_pred_1))

print('\n=== MLP with 5 Hidden Layers ===')
print('MAE:', mean_absolute_error(y_test, y_pred_5))
print('RMSE:', root_mean_squared_error(y_test, y_pred_5))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(X_train, y_train, '-b', label='Training Data')
plt.plot(X_test, y_test, '-r', label='Test Data')
plt.plot(X_test, y_pred_1, '-g', label='MLP One Hidden Layer')
plt.plot(X_test, y_pred_5, '-y', label='MLP Five Hidden Layers')
plt.legend()
plt.title('Training and Test Data with Predictions')
plt.xlabel('Months')
plt.ylabel('Sunspot Activity')
plt.show()