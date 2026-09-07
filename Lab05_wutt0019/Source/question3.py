"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question3.py
Date: 07-09-2026
Description:
"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import RocCurveDisplay

# Load in iris
iris = load_iris()
X = iris.data
y = iris.target

# Turn into binary classification
y = (y == 0).astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Model init and fitting
dtc = DecisionTreeClassifier(max_depth=3, min_samples_split=4, min_samples_leaf=2)
dtc.fit(X_train, y_train)

# Plot ROC curve
RocCurveDisplay.from_estimator(dtc, X_test, y_test)
plt.show()