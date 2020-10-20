# =============================================================================
# Kernel Support Vecto Machine in Python using Scikit-Learn
# Program wrote by Igor Melo, Octobre 2020
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

# Splitting Dataset into Train and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

# In this problem we need to employ feature scaling due to the scale difference between
# independent variables (Age and Salary) and dependent variable (Purchase: yes or no / 0 or 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training the SVM model to a non linear kernel

# This program is an other simple example of SVM, but here we change the kernel. The SVM
# method can be used for a linear and non-linear problems. In non-linear problems we define
# a hyper plan gave by a kernel function (Gaussian, polynomial, sigmoidal). The non-linear
# function makes a space transformation, from this transformation we are able to define 
a new delimiter that allow us to classify the possibles classes in the problem. In this model
the support vector will be gave by the hyper plan.


from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Here we utilize the class SVM and SVC object. The chosen parameters here
# are kernel = rbf (Gaussian function) and we fix the seed of random generator.
# For this object we can use also polynomial, sigmoidal and precomputed. Each
# kernel is appropriate to a kind of problem. The better way to know which kernel
# we will use is to try one by one.

# Making a Single Prediction

# As we already know, after training the model, we can make predictions.
# Attention (We need to employ Standardscaler because the entry values are Age and Salary in the normal scale)

print(classifier.predict(sc.transform([[30, 80000]])))

# Checking the test results

# One way to verify your results is to print the values predicted next to the test values

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)),1))

# Checkin matrix confusion and accuracy score

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
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# ===================================================================================================
# In this example we employed a non-linear SVM model. We chose the Gaussian kernel to solve
# the problem. The accuracy is very satisfactory. As we can see the curve traced by the trained model
# delimits two region in a specie of Gaussian curve. Here we can verify directly the relation between
# salary and purchase. The higher salary the greater the chance to purchase. The relation age and 
# purchase is also well defined and can be visualized. For example, someone that has 45 years old
# and a low salary can be a costumer. But the probability to someone that has a great salary and a
# high age to buy is more possible. This model is a powerful tool to construct a selection model.
# ===================================================================================================
