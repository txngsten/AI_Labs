

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Fetch MNIST, subsample for computability on my laptop :)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
rng = np.random.default_rng(42)
idx = rng.choice(len(mnist.data), size=10000, replace=False)
X, y = mnist.data[idx], mnist.target[idx]

# Train test slit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Pipeline for gridsearch
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))),
])

# Different solvers
param_grid = {'clf__estimator__solver': ['liblinear', 'lbfgs', 'saga']}

# Grid search with 5-fold cv
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

# Stats for each solver
for solver, mean, std in zip(
    param_grid['clf__estimator__solver'],
    gs.cv_results_['mean_test_score'],
    gs.cv_results_['std_test_score'],
):
    print(f'{solver:10s} {mean:.4f} (+/-{std:.4f})')

# Best solver
print('\nBest:', gs.best_params_, f'{gs.best_score_:.4f}')

# Visualize winner only
y_pred = gs.predict(X_test)
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(gs, X_test, y_test, cmap='Blues', colorbar=False)
plt.title(f"Best solver: {gs.best_params_['clf__estimator__solver']}")
plt.show()