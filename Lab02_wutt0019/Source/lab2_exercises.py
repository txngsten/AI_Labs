# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: lab2_exercises.py
# Date: 07-08-2026
# Description: Python file of the lab 2 exercises, mirrors the notebook

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

# Load in dataset
csv_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ['Pregnancies', 'Glucose', 'Blood pressure', 'Skin thickness','Insulin', 'Body mass index', 'Diabetes pedigree function', 'Age', 'Outcome']

pid_data = pd.read_csv(csv_url, header=None, names=col_names, engine='python')

# Take a look
print(pid_data.head())

# Summary statistics
print(pid_data.describe())

# Plotting the data
pid_data.hist()
plt.suptitle('Histogram of Input Features')
plt.tight_layout()
plt.subplots_adjust(top=0.90)  # room for suptitle
plt.show()

# Plotting pairwise relationships
sns.pairplot(pid_data, hue='Outcome')
plt.suptitle('Pairwise Relationships of Features')
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # room for suptitle
plt.show()

# Train-test split
X = pid_data.drop('Outcome', axis=1)
Y = pid_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fitting and testing prediction accuracy
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

dtc_predictions = dtc.predict(X_test)

accuracy = accuracy_score(y_test, dtc_predictions)
print(f'Accuracy for Predictions: {accuracy:.2f}')

# Use graphviz to visualize the decision tree model
os.environ['PATH'] += os.pathsep + '/opt/homebrew/Cellar/graphviz'

dot_data = export_graphviz(
    dtc,
    out_file=None,
    feature_names=X.columns,
    class_names=['No Diabetes', 'Diabetes'],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view()

# Display text representation of the decision tree model
tree_struct = export_text(dtc, feature_names=list(X.columns))
print("Decision Tree Structure:\n", tree_struct)

# Display features of importance
feature_importance = dtc.feature_importances_
feature_importance_summary = sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)

print('\n=== Feature Importance ===')
for feature, importance in feature_importance_summary:
    print(f'{feature}: {importance:.4f}')

# Grid search cross validation for finding optimal hyperparameters
hyperparam_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 10, 20, 30],
    'max_leaf_nodes': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [4, 10, 20, 50],
    'class_weight': [None, 'balanced'],
    'max_features': [None, 'sqrt', 'log2']
}

dtc_gs = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dtc_gs, param_grid=hyperparam_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyper-parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# See test set accuracy
gs_predict = grid_search.predict(X_test)
gs_accuracy = accuracy_score(y_test, gs_predict)

print(f"Grid Search Cross-Validation Accuracy: {gs_accuracy:.2f}")