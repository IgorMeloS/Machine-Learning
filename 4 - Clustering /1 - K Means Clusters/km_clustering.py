# =============================================================================
# K Means Clustering in Python using Scikit-learn
# Program wrote by Igor Melo, Octobre 2020
# This program has as objective to find the cluster regions based on the 
# dataset composed by five columns: CustumerID, Genre, Age, Anual Icome and
# Spending Score. We try to find regions with correlation between salary and
# spending score.
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('Mall_Customers.csv')

# For a pedagogical proposal we just use the two last columns Anual incoming and Spending Score.
# In this way we can visualize the clustering regions based on these two variables

X = dataset.iloc[:, [3,4]].values

# The K-Means Clustering is a method to distinguish possibles groups or cluster 
# in a data set. The method consists in to choose a certain number K of cluster.
# From this, we place a new data point as a centroid in each cluster. From the 
# distance between the centroids we define the cluster regions. After we replace 
# the centroids and we verify if the any data points change of cluster, in a positive 
# case, we recalculate a new centroids position, otherwise we finish our model. One drawback 
# of this method is the random trap. If we initialize the centroids randomly, we 
# can have a bad outcome, to avoid it we utilize k-means++. 
# Is there a method to choose the number of clusters? 
# Yes, we utilize the metric WCSS (Within-Cluster-Sum-of-Squares) also known as Elbow method. 
# We can plot WCSS x number of cluster, there is a region that seems an elbow, this 
# value is the optimal number of clusters to the data set. 

# Elbow Method to identify the apropriate number of clusters

from sklearn.cluster import KMeans

# The list of wcss values
wcss = []
for i in range(1, 11):
    kMeans = KMeans(n_clusters= i, init='k-means++', random_state= 42)
    kMeans.fit(X)
    wcss.append(kMeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the whole dataset

# Once time that we know the optimal number of clusters we train our model
# We utilze the class sklearn.cluster and the object KMeans.
# The argument init='k-means++' avoid the random initialization.

kMeans = KMeans(n_clusters= 5, init='k-means++', random_state= 42)

# Defining a new dependent variables, that are classifiers.

y_kmeans = kMeans.fit_predict(X)

# Visualizing the Clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'k', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'g', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'orange', label = 'Cluster 5')
plt.scatter(kMeans.cluster_centers_[:, 0], kMeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Some statical informations about each cluster (Salary and Spending Score)

cluster_0 = X[y_kmeans == 0]
cluster_1 = X[y_kmeans == 1]
cluster_2 = X[y_kmeans == 2]
cluster_3 = X[y_kmeans == 3]
cluster_4 = X[y_kmeans == 4]

cluster_0[0].mean()
cluster_0[0].std()
cluster_0[1].mean()
cluster_0[1].std()

cluster_1[0].mean()
cluster_1[0].std()
cluster_1[1].mean()
cluster_1[1].std()

cluster_2[0].mean()
cluster_2[0].std()
cluster_2[1].mean()
cluster_2[1].std()

cluster_3[0].mean()
cluster_3[0].std()
cluster_3[1].mean()
cluster_3[1].std()

cluster_4[0].mean()
cluster_4[0].std()
cluster_4[1].mean()
cluster_4[1].std()

# =================================================================================================
# Here we presented a simple example of clustering using K-Means Method. The algorithm is very
# intuitive and easy to use. The results can be visualized in the graphic and we can make some 
# interpretations as separate a possible costumer based on the groups. For example, if we want to
# select a possible good costumer based on the groups, the clusters 1, 2  and 3 are not interesting. 
# The first cluster is characterized to low salary and high spending score. The cluster 2 is on the 
# middle and it can be a problem in respect to the low salary and high  consummation. The cluster 3 
# has a good responsibility, low salary and low spending score. The clusters 4 and 5 present a high 
# salary. The cluster 4 presents a consummation proportional  to the salaries. The cluster 5 is very 
# interesting because we can see a strong salaries and a low spending score. Based on these results,
# to sale an expensive product the better choice is the custumers in the clusters 4 and 5.
# ==================================================================================================
