import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Fetch the car evaluation dataset
car = fetch_ucirepo(id=19)

# Get features and labels as pandas dataframes
X = car.data.features
y = car.data.targets

# Inspect variables
print(car.variables)

# One-hot encode
X = pd.get_dummies(X, drop_first=True)
y = y.values.ravel()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model init and fitting
lr = LogisticRegression(max_iter=10000, random_state=42)
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate model
print('Accuracy:', accuracy_score(y_pred, y_test))
print('Confusion Matrix:\n', confusion_matrix(y_pred, y_test))
