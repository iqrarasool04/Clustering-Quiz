import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

points = np.array([(2, 10), (2, 5), (8, 4), (5, 8), (7, 5), (6, 4), (1, 2), (4, 9)])
points = points.reshape(-1, 2)

#calculation of euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#clustering function
def k_means_clustering(points, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(points)
    #plotting data
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title(f'K-Means Clustering for k={k}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

#elbow method
def elbow_method(points, k_max):
    distortions = []
    K_range = range(1, k_max + 1)

    for k in K_range:
        if k <= len(points): 
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(points)
            distortions.append(kmeans.inertia_)

    #plotting elbow curve
    plt.plot(K_range[:len(distortions)], distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.show()

#specifying values
K_values = [2, 3]
k_max = 8
for k in K_values:
    k_means_clustering(points, k)
elbow_method(points, k_max)

