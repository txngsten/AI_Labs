# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: lab3_exercises.py
# Date: 18-08-2026
# Description: Python file of the lab 3 exercises, mirrors the notebook

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load in the dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Convert to dataframe and print head
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(iris_df.head())

# Summary statistics
print(iris_df.describe())

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plotting the clusters
iris_df['Cluster'] = labels
sns.pairplot(iris_df, hue='Cluster', palette='viridis', markers=['o', 's', 'D'])
plt.suptitle('Pair Plot of Iris Data-set with K-means Clustering', y=1.02)
plt.show()

# 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], iris_df['petal length (cm)'],
           c=iris_df['Cluster'], cmap='viridis', edgecolors='k', s=150)

ax.set_title('3D Scatter Plot of K-Means Clustering on Iris Dataset')
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')

plt.show()

# Find cluster centres
print(f'Cluster Centers:\n{kmeans.cluster_centers_}')

# Calculate interia
print(f'Inertia: {kmeans.inertia_}')

# Calculate silhouette score
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette_avg}')

# Calculate optimal value for k using silhouette scores
silhouette_scores = []
k_range = range(2, 11)

for k_clusters in k_range:
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Scores')
plt.xticks(k_range)
plt.grid(True)
plt.show()

print(f'Max Silhouette Score: {max(silhouette_scores)}')