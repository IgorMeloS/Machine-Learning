# =============================================================================
# Naïves Bayes Classification in Python using Scikit-Learn
# Program wrote by Igor Melo, Octobre 2020
# This program has as objective predict a possible costumer to a SUV model.
# The model is based in a dataset composed by Age, Salary, Purchase 
# (yes or no /0 or 1).
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset and Splitting into Independent and Dependent variables

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Dataset into Train and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

# In this case our independents variables are in a different scale, far from 0 or 1 (dependent variable).
# For this reason we transforms the independents variables to values around 0 and 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training our Naïves Bayes Classification Model

# The Naïve Bayes method consists in classifier the values by a given probability.
# The trick is to calculate the probability of a given value in reference to a class.
# For example, if we have a new data point X and two possibles classes we calculate 
# P(class 1 | X) = P(X | class 1) x P(class 1) / P(X) and 
# P(class 2 | X) = P(X | class 2) x P(class 2) / P(X) , after we compare the results and
# we choose the higher likelihood value.

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Here we utilize naive_bayes class and GaussianNB object

# Making a single prediction

# As we already know, after training the model, we can make predictions.
# Attention (We need to employ Standardscaler because the entry values are Age and Salary in the normal scale)

print(classifier.predict(sc.transform([[30, 80000]])))

# Checking test values

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

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
plt.title('Naive Bayes (Training set)')
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
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# ================================================================================================
# Here we employed a Naïve Bayes Classifier. We can see that the method gives a good accuracy.
# But the drawback of this method is the consideration of independence between the variables.
# In the graphic we can see the curve that delimits two regions (red and green). Once again 
# we can see the relation between salary and purchase, high salary is more susceptibility to buy.
# And also we can see that the relation age purchase is lass decisive in the choice.
# ================================================================================================
