"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question2.py
Date: 07-09-2026
Description:
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load in iris
iris = load_iris()
X = iris.data
y = iris.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Model init and fitting
dtc = DecisionTreeClassifier().fit(X_train, y_train)

# Make predictions
y_pred = dtc.predict(X_test)

# Print metrics
print(
    'Classification Report:\n',
    classification_report(y_test, y_pred, target_names=iris.target_names)
)