# =============================================================================
# Logistic Regression in Python using Scikit-Learn
# Program wrote by Igor Melo, Octobre 2020.
# The goal of this program is to train a classification model to predict a possible
# SUV purchase.
# The data set is composed by tree columns that contains Age, Salary, Purchase
# (yes or no / 0 or 1)
# =============================================================================

# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

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

# Training our Logistic Regression Model

# This linear model uses a sigmoidal function to obtain the density of probability 
# appropriate to the function Y (can be many kind of function). Once we have the 
# density of probability we can classify the classes by the probability, for example,
# if we have a low probability (< 0.5) the response can be classified as no, otherwise yes.
    
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# We utilizes the classe linear_model and the object LogisticRegression, the only
# argument that we choose is random_state, but you can see other parameters in API.

# Making a single prediction 

# As we already know, after training the model, we can make predictions.
# Attention (We need to employ Standardscaler because the entry values are Age and Salary in the normal scale)

xc = classifier.predict(sc.transform([[30, 120000]]))
if xc == 0:
    print('Non, ce client n est pas un possible acheteur, passez a une autre chose')
else :
    print('Oui, ce client est un possible acheteur, contuniez.')

# Testing and visualizing the predicted results

# One way to verify your results is to print the values predicted next to the test values

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Matrix confusion and accuracy score

# As we can see in the last print, our model presents good previsions, but we can find bad results.
# Is important to know how efficient is our model, to do it, we can utilize matrix confusion and accuracy score
# In the first we are going to see a matrix that contains the number of good and bad predictions referent to each class.
# In the second tool we have the ratio of good prevision called accuracy score.

from sklearn.metrics import confusion_matrix, accuracy_score
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print(accuracy_score(y_test, y_pred))

# Visualising the Training set results

# In this problem we can visualize the results because we have two independent variables.
# Problems with more than two independent variables, we do not have this auxiliary tool

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
plt.title('Logistic Regression (Training set)')
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
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# =============================================================================
# This program is a simple example of Logistic Regression using Sci-kit-Learn to
# classify a possible costumer to a SUV model. As we can see, the linear model draw a straight
# line that separates the graphic in two regions, red and green (0 and 1, respectively). The model
# has a good accuracy, but this is not the better model to analysis this kind of problem. The curve
# has a considerable slope that imposes some problems, for example, someone that have 20 years
# old and a higher salary 120000 dollars can be a possible costumer, due to the relation between 
# salary and decision, but in this problem this costumer will not to buy. Other problem interpretation
# is someone with 50 years old and low salary as a costumer, this indicates that age is more decisive.  
# =============================================================================
