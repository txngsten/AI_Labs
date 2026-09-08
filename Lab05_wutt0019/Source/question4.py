"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question4.py
Date: 07-09-2026
Description: Computing MSE, MAE, and RMSE on for decision tree regressor on the California housing dataset.
"""


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

# Fetch dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Model init and fitting
dtr = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
).fit(X_train, y_train)

# Make predictions
y_pred = dtr.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

# Print metrics
print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)
print('Root Mean Squared Error: ', rmse)