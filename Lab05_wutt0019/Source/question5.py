"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question5.py
Date: 08-09-2026
Description: Comparing a fully fitted decision tree regressor and a optimally pruned one on the California housing dataset.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# Fetch dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Train test validation split, train size 60%, test size 20%, val size 20%
X_train, X_test_val, y_train, y_test_val = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_test_val,
    y_test_val,
    test_size=0.5,
    random_state=42
)

# Fully fitted decision tree
dtr = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# Optimally pruned
path = dtr.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas
alphas = alphas[alphas > 0]
alphas = np.geomspace(alphas.min(), alphas.max(), 250)

gs_cv = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    {'ccp_alpha' : alphas},
    cv=5,
    n_jobs=-1
).fit(X_train, y_train)

# Pick best pruned tree
pruned = gs_cv.best_estimator_

# Make predictions
y_pred_full = dtr.predict(X_test)
y_pred_pruned = pruned.predict(X_test)

"""
Computes and prints metrics for two prediction sets,
pruned and full fitted Decision Tree Regressors.
"""
def compute_and_print(y_test, y_pred_pruned, y_pred_full):
    # Compute metrics
    pruned_r2 = r2_score(y_test, y_pred_pruned)
    pruned_rmse = root_mean_squared_error(y_test, y_pred_pruned)
    full_r2 = r2_score(y_test, y_pred_full)
    full_rmse = root_mean_squared_error(y_test, y_pred_full)

    # Print metrics
    print(f'Pruned R^2 Score: {pruned_r2} and Root Mean Squared Error {pruned_rmse}')
    print(f'Full Fitted R^2 Score: {full_r2} and Root Mean Squared Error {full_rmse}')

# Compute and print metrics, on test set
print('=== Test Set Metrics ===')
compute_and_print(y_test, y_pred_pruned, y_pred_full)

# Make predictions on hold-out validation set
y_pred_full_val = dtr.predict(X_val)
y_pred_pruned_val = pruned.predict(X_val)

# Compute and print metrics, on validation set
print('\n=== Validation Set Metrics ===')
compute_and_print(y_val, y_pred_pruned, y_pred_full)





