
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, RocCurveDisplay

# Fetch dataset
ionosphere = fetch_ucirepo(id=52)

# Load into pandas dataframes
X = ionosphere.data.features
y = (ionosphere.data.targets.values.ravel() == 'g').astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model init and fitting
lr = LogisticRegression(solver='liblinear', l1_ratio=1, C=0.1, max_iter=10000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Predict and score using ROC AUC Score
y_pred = lr.predict(X_test_scaled)
roc_score = roc_auc_score(y_test, y_pred)
print('ROC AUC Score:', roc_score)

# Plot ROC curve
RocCurveDisplay.from_estimator(lr, X_test_scaled, y_test)
plt.show()