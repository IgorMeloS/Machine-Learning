# =============================================================================
# Hierarchical Clustering in Python using Scikit-Learn
# Program wrote by Igor Melo, Octobre 2020.
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

# The Hierarchical Clustering is a method to classify cluster regions. Differently to K-Means method,
# here we consider all N data points like a clusters. To have a full model model we need to interact 
# all data points in many steps. In the first step we choose the two nearest (Euclidean distance) data 
# point and we consider a new cluster, in this way we have N-1 cluster. We make the same process until
# we have just one cluster. How choose we the number of clusters? We plot a dendrogrammed and from the
# graphic we can visualize the optimal number of cluster. In a situation where we have more than two
# closest cluster to a data point we look for the cluster with lower variance value

# Making a dendrograme to have the optimal number of clusters

import scipy.cluster.hierarchy as sch
dendrograme = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# To know the optimal number of clusters, in the graphic we look for the vertical lines that reach two 
# horizontal lines. We select the vertical line that has highest distance. After we split this vertical 
# line in the center with a horizontal line. After we visualize the number of vertical lines that split 
# line reached, this is the optimal number of clusters

# Training the Hierarchical Clustering Model

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

# We use the cluster object and AgglomerativeClustering class. For this example we choose 
# affinity parameter as Euclidean, but there exist others metrics. The parameter linkage 
# is ward (minimal variance).

# Defining a new dependent variables, that are classifiers.

y_hc = hc.fit_predict(X)

# Visualizing the clusters 
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'k', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 50, c = 'g', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 50, c = 'orange', label = 'Cluster 5') 
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Some statical informations about each cluster (Salary and Spending Score)

cluster_0 = X[y_hc == 0]
cluster_1 = X[y_hc == 1]
cluster_2 = X[y_hc == 2]
cluster_3 = X[y_hc == 3]
cluster_4 = X[y_hc == 4]

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

# ==================================================================================================
# Here we presented a simple example of clustering using Hierarchical Clustering. The results can 
# be visualized in the graphic and we can make some interpretations as separate a possible costumer 
# based on the groups. For example, if we want to select a possible good costumer based on the groups, 
# the clusters 1, 2  and 3 are not interesting. 
# The first cluster is characterized to low salary and high spending score. The cluster 2 is on the 
# middle and it can be a problem in respect to the low salary and high  consummation. The cluster 3 
# has a good responsibility, low salary and low spending score. The clusters 4 and 5 present a high 
# salary. The cluster 4 presents a consummation proportional  to the salaries. The cluster 5 is very 
# interesting because we can see a strong salaries and a low spending score. Based on these results,
# to sale an expensive product the better choice is the custumers in the clusters 4 and 5.
# ===================================================================================================