# =============================================================================
# Random Forest Classification in Python using Scikit-Learn
# This program is a simple example to demonstrate how to apply a classification
# model using Random Forest Classifier algorithm. The goal is to predict two
# possible classes of customers, they that will purchase or not a SUV model.
#
# **Dataset Description**
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
y = dataset.iloc[:, -1].values # Dependent Variables

# Splitting Dataset into Training and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# =============================================================================
# Building the model - Random Forest Classifier
# 
# **Definition from Scikit-Learn website**
# 
# The goal of ensemble methods is to combine the predictions of several base
# estimators built with a given learning algorithm in order to improve
# generalizability / robustness over a single estimator.
# 
# Two families of ensemble methods are usually distinguished:
# 
# +      In **averaging methods**, the driving principle is to build several
#        estimators independently and then to average their predictions. On
#        average, the combined estimator is usually better than any of the
#        single base estimator because its variance is reduced.
# 
#     Examples: Bagging methods, Forests of randomized trees, …
# 
# +    By contrast, in **boosting methods**, base estimators are built
#      sequentially and one tries to reduce the bias of the combined estimator.
#      The motivation is to combine several weak models to produce a powerful
#      ensemble.
# 
#     Examples: AdaBoost, Gradient Tree Boosting, …
#
# Here, we consider the **Random Forest** 
# In random forests, each tree in the ensemble is built from a sample drawn
# with replacement (i.e., a bootstrap sample) from the training set.
# 
# Furthermore, when splitting each node during the construction of a tree, the
# best split is found either from all input features or a random subset of size
# max_features.
# 
# The purpose of these two sources of randomness is to decrease the variance of
# the forest estimator. Indeed, individual decision trees typically exhibit
# high variance and tend to overfit. The injected randomness in forests yield
# decision trees with somewhat decoupled prediction errors. By taking an
# average of those predictions, some errors can cancel out. Random forests
# achieve a reduced variance by combining diverse trees, sometimes at the cost
# of a slight increase in bias. In practice the variance reduction is often
# significant hence yielding an overall better model.
# 
# From Scikit-Learn library we have a class called Ensemble that contains the
# object RandomForestClassifier.
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

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

### Visualising the results

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
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the test set results

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

# Conclusion
# =============================================================================
# In this program we implemented a simple example of a Random Tree
# Classification. This non-linear method has a good accuracy, better than
# Decision Tree, it proves that ensembles methods can be a powerful tool. In
# the plot we can see two possibles regions green and red. The first predict a
# possible costumer and the second, someone which does not want to buy a SUV
# model. In the axis $x$ we have age and in axis $y$ salary. There is a
# relation between age and salary, for someone that has 50 years old and salary
# 120000 dollars, we find this person in the green region, higher salary is
# deterministic. Other important aspect of this model is the fact that someone
# which has 20 years old, but a higher salary, can be a possible costumer, in
# the linear models this is not possible. The drawback presented for this model
# is the interpretation of some little regions and points out of your range
# colors.
# =============================================================================

