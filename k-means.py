import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as metric


def initialize(dataset, k):
    # Create random cluster centroids for intial

    # Find number of features in dataset
    n = np.shape(dataset)[1]

    # The centroids
    centroids = np.mat(np.zeros((k, n)))

    # Fill list with random points
    for i in range(n):
        min_i = min(dataset[:, i])
        range_i = float(max(dataset[:, i]) - min_i)
        centroids[:, i] = min_i + range_i * np.random.rand(k, 1)

    # Return centroids
    return centroids


def cluster(dataset, k):
    # The clustering algorithm

    # Number of rows in dataset
    m = np.shape(dataset)[0]

    # Assign cluster classes
    cluster_class = np.mat(np.zeros((m, 2)))

    # Initialize centroids
    centroids = initialize(dataset, k)

    # Report original centroids
    centroids_origin = centroids.copy()

    changed = True
    n_iter = 0

    # Loop until no changes to cluster assignments
    while changed:

        changed = False

        # For every row in dataset
        for i in range(m):

            # Track minimum distance, and index it
            min_dist = np.inf
            min_index = -1

            # Find distances
            for j in range(k):
                distance = metric.euclidean(centroids[j, :], dataset[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j

            # Check if a point's cluster assignment has changed
            if cluster_class[i, 0] != min_index:
                changed = True

            # Assign point to nearest cluster and distance
            cluster_class[i, :] = min_index, min_dist ** 2

        # Update centroid locations
        for cent in range(k):
            points = dataset[np.nonzero(cluster_class[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(points, axis=0)

        # iterations
        n_iter += 1
        print(n_iter)
    # Return stuff when done
    return centroids, cluster_class, n_iter, centroids_origin


# Dataset preparations
print('k Means Clustering Algorithm in Python')
filename = 'kmeans_data.csv'
dataset = np.genfromtxt(filename, delimiter=',')
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.show()

# Set Number of Centroids
var = int(input("Number of Centroids: "))
k = var

# Perform k-means clustering
centroids, cluster_classes, n, origin_centroids = cluster(dataset, k)

# Output results
print('Number of iterations:', n)
print('\nFinal centroids:\n', centroids)
print('\nOriginal centroids:\n', origin_centroids)
newSet = np.concatenate((dataset, cluster_classes), axis=1)
plt.scatter([newSet[:, 0]], [newSet[:, 1]], c=[newSet[:, 2]])
plt.show()
