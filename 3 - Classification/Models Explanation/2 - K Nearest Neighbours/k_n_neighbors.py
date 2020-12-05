# =============================================================================
# K Nearest Neighbors in Python Using Scikit-Learn
# Program wrote by Igor Melo, Octobre 2020.
# This program has as objective predict a possible costumer to a SUV model.
# The model is based in a dataset composed by Age, Salary, Purchase 
# (yes or no /0 or 1).
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset and Splitting Independent and Dependent Variables

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spliting the dataset into Train and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

# In this problem we need to employ feature scaling due to the scale difference between
# independent variables (Age and Salary) and dependent variable (Purchase: yes or no / 0 or 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training our KNN model

# KNN is a non-linear model and very simple to understand and employ it. When we have
# a set of points that can be divided  by classes, if we want to know which class a new data 
# point will be, we choose a number of neighbors near to this new data point and we count the
# number of points in each category. For example, if we have two classes 1 and 2, the new data
# point have tree neighbors class 1 and two neighbors class 2, the algorithm will classify this new
# data point in class 1. The distance between the point can be made in many ways, it’s depends on 
# metric choosing. In almost all cases we utilize the Minkowski’s metric that uses Euclidean
# distance or Manhattan distance

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Here we utilize Neighbors class and we call the object KNeighborsClassifier
# The n_neighbors is 5, the chosen metric is Minkowski and p = 2 is a stadart
# Euclidean distance (p = 1 - Manhattan distance)


# Making a single prediction

# As we already know, after training the model, we can make predictions.
# Attention (We need to employ Standardscaler because the entry values are Age and Salary in the normal scale)

print(classifier.predict(sc.transform([[30, 80000]])))

# Testing the test set values

# One way to verify your results is to print the values predicted next to the test values

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)),1))

# Checking Confusion Matrix and Accuracy Score

# As we can see in the last print, our model presents good previsions, but we can find bad results.
# Is important to know how efficient is our model, to do it, we can utilize matrix confusion and accuracy score
# In the first we are going to see a matrix that contains the number of good and bad predictions referent to each class.
# In the second tool we have the ratio of good prevision called accuracy score.

from sklearn.metrics import confusion_matrix, accuracy_score
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print(accuracy_score(y_test, y_pred))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# =============================================================================
# Here we presented a simple example of KNN model. This non-linear model present a good
# accuracy. Due to the non-linearity of this problem we can visualize in the graphic two distinct
# regions. Green to 1 and red to 0, the line that divides is not linear, for this reason we can see
# more easily the relation between salary and purchase. In this method for example, someone that
# has 20 years and a high salary 120000 can be a classified as a costumer. The relation between
# age and purchase is lass expressive, but we also see that more old is the person it becomes more
# susceptible to buy it. In the graphics we can also see a little red region in the region predominately 
# green, it is complicated to give an interpretation. This model is very efficient and give a good 
# accuracy, the single drawback is make the choice of number neighbors. 
# =============================================================================
