from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def standardize_data(df):
    """Standardize data for clustering."""
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def perform_kmeans(data, n_clusters):
    """Apply K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    return kmeans.fit_predict(data)

def find_optimal_clusters(data, max_k):
    """Use the elbow method to find the optimal number of clusters."""
    wcss = []
    for i in range(1, max_k+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss
