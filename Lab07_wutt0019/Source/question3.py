"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: questions3.py
Date: 29-09-2026
Description: Fitting MLP models with different solvers/optimizers.
"""

# Imports
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training models with different optimizers
results = {}
solvers = ['lbfgs', 'sgd', 'adam']

for solver in solvers:
    mlp = MLPRegressor(
        hidden_layer_sizes=(50, 50, 50),
        activation='relu',
        solver=solver,
        max_iter=10000,
        random_state=42
    ).fit(X_train_scaled, y_train)

    # Make predictions
    results[solver] = mlp.predict(X_test_scaled)

# Compute metrics
for solver in solvers:
    print(f'=== Results for {solver} solver ===')
    print('MAE:', mean_absolute_error(y_test, results[solver]))
    print('RMSE:', root_mean_squared_error(y_test, results[solver]))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(X_train, y_train, '-b', label='Training Data')
plt.plot(X_test, y_test, '-r', label='Test Data')
plt.plot(X_test, results['lbfgs'], '-g', label='MLP with LBFGS Solver')
plt.plot(X_test, results['sgd'], '-y', label='MLP with Stochastic Gradient Descent Solver')
plt.plot(X_test, results['adam'], '-y', label='MLP with Adam Solver')
plt.legend()
plt.title('Training and Test Data with MLP Predictions')
plt.xlabel('Months')
plt.ylabel('Sunspot Activity')
plt.show()