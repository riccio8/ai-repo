from sklearn.cluster import KMeans, DBSCAN

# KMeans
data = [[1, 2], [2, 3], [8, 9], [9, 10]]
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
print(kmeans.labels_)

# DBSCAN
dbscan = DBSCAN(eps=2, min_samples=2)
dbscan.fit(data)
print(dbscan.labels_)
