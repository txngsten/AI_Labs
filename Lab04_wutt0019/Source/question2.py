
import numpy as np

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Import iris dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.1, random_state=42
)

# Model init
lr = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)

# Cross-validation scoring of training set, K = 5
cross_val_scores = cross_val_score(lr, X_train, y_train, cv=5)

# Show results
print('Cross-Validation Scores:', cross_val_scores)
print('Mean CV Score:', np.mean(cross_val_scores))

