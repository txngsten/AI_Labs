"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question1.py
Date: 07-09-2026
Description: Shows confusion matrix for logistic regression on iris dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load in iris
iris = load_iris()
X = iris.data
y = iris.target

# Group Virginica and Versicolour together as single class
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
lr = LogisticRegression(max_iter=10000, random_state=42).fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Negative', 'Predicted Positive'],
    yticklabels=['Actual Negative', 'Actual Positive']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()