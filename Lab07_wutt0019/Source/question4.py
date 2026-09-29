"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: questions4.py
Date: 29-09-2026
Description: Fitting MLP models with GridSearchCV across model hyperparameters.
"""

# Imports
import numpy as np
import statsmodels.api as sm

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Scale data
scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# Fit the unoptimized MLP model
mlp_unoptimized = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
).fit(X_train_scaled, y_train_scaled)

# Perform grid search cross-validation on hyperparameter combinations
param_grid = {
    'solver': ['lbfgs', 'adam', 'sgd'],
    'batch_size': [64, 128, 256],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.01, 0.001, 0.0001]
}

mlp_optimized = GridSearchCV(
    MLPRegressor(
        hidden_layer_sizes=(50, 50),
        max_iter=2000,
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
).fit(X_train_scaled, y_train_scaled)

# Show best params
print(mlp_optimized.best_params_)

# Make predictions
y_pred_unoptimized = mlp_unoptimized.predict(X_test_scaled)
y_pred_optimized = mlp_optimized.predict(X_test_scaled)

# Compute and print metrics
print('=== Unoptimized MLP Model ===')
print('MAE:', mean_absolute_error(y_test_scaled, y_pred_unoptimized))
print('RMSE:', root_mean_squared_error(y_test_scaled, y_pred_unoptimized))

print('\n=== Optimized MLP Model ===')
print('MAE:', mean_absolute_error(y_test_scaled, y_pred_optimized))
print('RMSE:', root_mean_squared_error(y_test_scaled, y_pred_optimized))