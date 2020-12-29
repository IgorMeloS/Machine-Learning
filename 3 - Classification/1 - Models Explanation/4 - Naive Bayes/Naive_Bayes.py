# =============================================================================
# Naïves Bayes Classification in Python using Scikit-Learn
# This program is a simple example to demonstrate how to apply a classification
# model using Na6ives Bayes algorithm. The goal is to predict two possible
# classes of customers, they that will purchase or not a SUV model.
#
# **Dataset Description**
#
# The dataset is composed by 3 columns and 400 rows. The feature (columns) are
# Age, Estimated Salary and Purchases (target variable 0 or 1). Each row
# represents one customer.
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values # Independent Variables
y = dataset.iloc[:, -1].values # Dependent Variable

# Splitting Dataset into Train and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

### Building the Model - Naïves Bayes

## **Definition from Scikit-Learn website**

#Naive Bayes methods are a set of supervised learning algorithms based on applying
# Bayes’ theorem with the “naive” assumption of conditional independence between
# every pair of features given the value of the class variable. Bayes’ theorem
# states the following relationship, given class variable and dependent feature vector $x_{1}$ through $x_{n}$:
# $P(y|x_{1},...,x_{n}) = \frac{P(y)P(x_{1},...,x_{n}|y)}{P(x_{1},...,x_{n})}$.
# Using the naive conditional independence assumption that
# $P(x_{i}|y,x_{1},...,x_{i-1},x_{i+1},...,x_{n}) = P(x_{i}|y)$,
# for all $i$, this relationship is simplified to
# $P(y|x_{1},...,x_{n}) = \frac{P(y) \prod_{i=0}^{n} P(x_{i}|y)}{P(x_{1},...,x_{n})}$.
# Since $P(x_{1},...,x_{n})$ is constant given the input, we can use the following classification rule:
# $P(y|x_{1},...,x_{n}) \propto P(y)\prod_{i=0}^{n} P(x_{i}|y)$.
# The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of
#$ P(x_{i}|y)$.
#From Scikit-Learn library we have a class called Naives Bayes that contains the object GaussianNB.

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Training the model

classifier.fit(X_train, y_train)

# Making a single prediction

print(classifier.predict(sc.transform([[30, 80000]])))

# Predicting new results

y_pred = classifier.predict(X_test)

# Metrics (accuracy and confusion matrix)

from sklearn.metrics import confusion_matrix, accuracy_score
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print(accuracy_score(y_test, y_pred))

## Visualizing the results

# Visualizing the training set results


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

# Visualizing the test set results

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

# Conclusion
# =============================================================================
# Here, we employed a Naïve Bayes Classifier. We can see that the method gives
# a good accuracy, considering the independece among the features. The drawback
# of this method is the consideration of independence between the variables, in
# problems more complex it can not give a good results. 
#
# In the graphic we can see the curve that delimits two regions (red and green).
# Once again we can see the relation between salary and purchase, high salary is
# more susceptibility to buy. And also we can see that the relation age
# purchase is lass decisive in the choice.
# =============================================================================

