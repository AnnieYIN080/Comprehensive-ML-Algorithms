# R base
```
# Generate simulated data (similar to Python's make_blobs)
set.seed(42)
n_samples <- 300
n_features <- 2
n_clusters <- 3

# Generate the center points of three Gaussian distributions
centers <- matrix(c(2, 2, -2, -2, 2, -2), ncol=2, byrow=TRUE)
X <- matrix(0, nrow=n_samples, ncol=n_features)
for (i in 1:n_clusters) {
  idx <- sample(1:n_samples, n_samples/n_clusters)
  X[idx,] <- matrix(rnorm(length(idx)*n_features, mean=centers[i,], sd=0.8), ncol=n_features)
}

# run kmeans
kmeans_result <- kmeans(X, centers=n_clusters, nstart=25)

# Draw the clustering results
plot(X[,1], X[,2], col=kmeans_result$cluster, pch=19)
points(kmeans_result$centers[,1], kmeans_result$centers[,2], pch=8, cex=2, col="red")

```

# R with reticulate
```
install.packages("reticulate")
library(reticulate)

py_install("scikit-learn")
py_install("matplotlib")

np <- import("numpy")
plt <- import("matplotlib.pyplot")
sklearn_cluster <- import("sklearn.cluster")
sklearn_datasets <- import("sklearn.datasets")

# Generate simulated data
n_samples <- 300L
n_features <- 2L
n_clusters <- 3L
make_blobs <- sklearn_datasets$make_blobs
X_y <- make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42L)
X <- X_y[[1]]
y <- X_y[[2]]

# run KMeans
kmeans <- sklearn_cluster$KMeans(n_clusters=n_clusters)
kmeans$fit(X)

# Obtain the clustering results and center points
labels <- kmeans$labels_
centers <- kmeans$cluster_centers_

# !!! Draw the clustering results (drawing with R)
# The default backend used by matplotlib in the R/reticulate environment is not an interactive backend (cannot see the plot directly)
plot(X[,1], X[,2], col=as.factor(labels), pch=19)
points(centers[,1], centers[,2], pch=8, cex=2, col="red")

```
# Python
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate simulation data
n_samples = 300
n_features = 2
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Cluster using the KMeans algorithm
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

# Obtain the clustering results and clustering centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Draw the original data and the clustering results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='red')
plt.show()

```

