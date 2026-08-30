"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: question4.py
Date: 30-08-2026
Description: Logistic regression with class balancing on unbalanced wine dataset.
"""

import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, PrecisionRecallDisplay

# Import wine dataset
wine = fetch_ucirepo(id=186)

X = wine.data.features
y = (wine.data.targets.values.ravel() >= 7).astype(int)   # ~20% positive

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model init and fitting
lr = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Log loss
print('Log Loss:', log_loss(y_test, lr.predict_proba(X_test_scaled)))

# Plot precision recall curve
y_score = lr.predict_proba(X_test_scaled)[:, 1]
PrecisionRecallDisplay.from_estimator(lr, X_test_scaled, y_test)
plt.show()