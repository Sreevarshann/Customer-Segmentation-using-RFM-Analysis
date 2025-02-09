from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load segmented RFM data
rfm_segmented = pd.read_csv('data/segmented_rfm.csv')

# Normalize the data
rfm_normalized = (rfm_segmented[['Recency', 'Frequency', 'Monetary']] - rfm_segmented[['Recency', 'Frequency', 'Monetary']].mean()) / rfm_segmented[['Recency', 'Frequency', 'Monetary']].std()

# Determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(rfm_normalized)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(rfm_normalized)
rfm_segmented['Cluster'] = clusters

rfm_segmented.to_csv('data/final_clusters.csv', index=False)
