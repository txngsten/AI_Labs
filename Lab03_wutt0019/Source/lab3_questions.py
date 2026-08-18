# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: lab3_questions.py
# Date: 18-08-2026
# Description: Python file of the lab 3 questions, mirrors the notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mode

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, accuracy_score

# Load in dataset
wine = datasets.load_wine()
X = wine.data
Y = wine.target

# Dataframe conversion
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
print(wine_df.head())

# Summary statistics
print(wine_df.describe())

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plot the clusters
wine_df['Cluster'] = labels
sns.pairplot(wine_df, hue='Cluster', palette='viridis', markers=['o', 's', 'D'])
plt.suptitle('Pair Plot of Wine Data-set with K-means Clustering', y=1.02)
plt.show()

# Compute silhouette score
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette_avg}')

# Find optimal clusters using elbow method
s_scores = []
k_to_score = {}
k_range = range(2, 11)

# Compute the silhouette scors for each k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    silhouette_avg = silhouette_score(X, kmeans.labels_)

    s_scores.append(silhouette_avg)
    k_to_score[k] = silhouette_avg

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, s_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Scores')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Show max silhouette score and corresponding value of k
best_k = max(k_to_score, key=k_to_score.get)
print(f"Max Silhouette Score: {k_to_score[best_k]} with k={best_k}")

# Plotting the new clusters with optimal k
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plot the clusters
wine_df['Cluster'] = labels
sns.pairplot(wine_df, hue='Cluster', palette='viridis')
plt.suptitle('Pair Plot of Wine Data-set with K-means Clustering', y=1.02)
plt.show()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

# Silhouette score to find optimal k
s_scores = []
k_range = range(2, 11)

# Compute the silhouette scors for each k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    silhouette_avg = silhouette_score(X, kmeans.labels_)

    s_scores.append(silhouette_avg)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, s_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Scores')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Fit Kmeans on training data with optimal k from elbow curve above
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# Map clusters to classes
mapping = {
    c: mode(y_train[kmeans.labels_ == c], keepdims=False).mode for c in range(2)
}

# Show accuracy
pred = np.array([mapping[c] for c in kmeans.predict(X_test)])
print(f'Accuracy Score: {accuracy_score(y_test, pred)}')

# Feature scaling methods
scalers = {
    'None' : None,
    'standard' : StandardScaler(),
    'min_max' : MinMaxScaler()
}

# Different initialization methods
inits = ['k-means++', 'random']

# Compute results
results = []
for name, func in scalers.items():
    X_scaled = X if func is None else func.fit_transform(X)
    for init in inits:
        kmeans = KMeans(n_clusters=2, init=init, random_state=42)
        kmeans.fit(X_scaled)

        results.append({
            'Scaler': name,
            'Init Method': init,
            'Silhouette Score': silhouette_score(X_scaled, kmeans.labels_),
            'Inertia': kmeans.inertia_
        })

# Sort by silhouette score
sorted(results, key=lambda d: d['Silhouette Score'])

# Show results
for result in results:
    print(f'Scaler: {result['Scaler']}')
    print(f'Init Method: {result['Init Method']}')
    print(f'Silhouette Score: {result['Silhouette Score']}')
    print(f'Inertia: {result['Inertia']}\n')


# Pair plot of winner
kmeans = KMeans(n_clusters=2, init=results[0]['Init Method'], random_state=42)
X_scaled = X if results[0]['Scaler'] == 'None' else scalers[results[0]['Scaler']].fit_transform(X)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Plot the clusters
wine_df['Cluster'] = labels
sns.pairplot(wine_df, hue='Cluster', palette='viridis')
plt.suptitle('Pair Plot of Wine Data-set with K-means Clustering', y=1.02)
plt.show()