import seaborn as sns

# Load data with clusters
clustered_data = pd.read_csv('data/final_clusters.csv')

# Plot RFM distributions by cluster
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
sns.boxplot(x='Cluster', y='Recency', data=clustered_data)
plt.subplot(1, 3, 2)
sns.boxplot(x='Cluster', y='Frequency', data=clustered_data)
plt.subplot(1, 3, 3)
sns.boxplot(x='Cluster', y='Monetary', data=clustered_data)
plt.show()

# Profile each cluster
for cluster in clustered_data['Cluster'].unique():
    print(f"Profile for Cluster {cluster}")
    profile = clustered_data[clustered_data['Cluster'] == cluster].mean()
    print(profile)
