# =============================================================================
# Random Forest Classification in Python using Scikit-Learn
# Program wrote by Igor Melo, Octobre 2020
# This program has as objective predict a possible costumer to a SUV model.
# The model is based in a dataset composed by Age, Salary, Purchase 
# (yes or no /0 or 1).
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset and Splitting into Independent and Dependent Variables

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

# Training our Random Forest Classification Model

# Random Forest Classification is a non-linear model that seems Random Forest Regression.
# The method consists in take an average over a certain number of tree (Decision Tree Model), 
# each tree are placed in a certain value randomly, but we choose the number of trees.
# This method can be more precisely than Decision Tree Classification, bur more complicate to
# to make considerations.

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# In the object Random Forest Classifier we have many arguments that can improve our model
# Here choose n_estimator, criterion and random state. The first choose the number of threes, 
# the second the mathematical method to take account and the last fix a seed for the random 
# distribution. (There is others options that can be verified in scikit-learn web site)

# Making a single prediction

# As we already know, after training the model, we can make predictions.
# Attention (We need to employ Standardscaler because the entry values are Age and Salary in the normal scale)

print(classifier.predict(sc.transform([[30, 80000]])))

# Checking the test values

# One way to verify your results is to print the values predicted next to the test values

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_pred), 1)), 1))

# Checking confusion matrix and accuracy score

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
plt.title('Random Forest Classification (Training set)')
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
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# =====================================================================================================
# In this program we saw a simple example of a Random Tree Classification. This non-linear method
# has a good accuracy. In the plot we can see two possibles regions green and red. The first
# predict a possible costumer and the second someone that do not want to buy. In the axis x we have
# age and in axis y salary. There is a relation between age and salary, for someone that have 50 years
# old and salary 120000 dollars, we find him in the green region, higher salary is deterministic. 
# We can see that someone that have 20 years old, but a higher salary, can be a possible costume, 
# in the linear models this is not possible. But there red points in the region  predominantly green 
# and vice and versa. These points there are no much sense and are complicated to make a good 
# interpretation, this the drawback of method. 
# ======================================================================================================
