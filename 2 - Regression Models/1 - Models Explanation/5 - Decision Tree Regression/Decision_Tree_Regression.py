# =============================================================================
# Decision Tree Regression in Python using Scikit-Learn
#
# **Dataset Description**
# The dataset to this model is composed by three columns and 1o row. We have one
# feature in the second column, Level Position. Our response is the last column,
# Salary. Based on the level, we construct a polynomial regression model to
# predict the salary for a given level position.
# =============================================================================

## Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv') # Reading and Creating the dataframe
X = dataset.iloc[:, 1:-1].values # Independent variable.
y = dataset.iloc[:, -1].values # Dependent Variable

## Building the model - Decision Tree Regression

# Decision Trees (DTs)** are a non-parametric supervised learning method used for
# classification and regression. The goal is to create a model that predicts the
# value of a target variable by learning simple decision rules inferred from the
# data features.
# In this model we define a target variable, the decisions will toke referring
# this variable. For example, we have two axis $x$ and $y$, we choose $x=5$,
# the model separates two regions $x>5$ and $x<5$, from this region we can define
# other targets or make a decision.
# One of advantage of this method resides in the fact that we can consider
# categorical data and we do not need to perform a feature scaling.
# In Scikit-Learn we have the class Tree (that includes CART - Classification Regression Trees).

from sklearn.tree import DecisionTreeRegressor # Class and object
DCR = DecisionTreeRegressor(random_state=0) # Creating the object
DCR.fit(X,y) # Fitting and traning the model

### Making a Single Prediction

print(DCR.predict([[7]]))

### Visualizing the Results

X_grid = np.arange(min(X), max(X), 0.001) # Increasing the number of points
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red') # Scatter Plot
plt.plot(X_grid, DCR.predict(X_grid)) # Decision Tree Plot
plt.title('Decision Tree Regression')
plt.xlabel('Level Position')
plt.ylabel('Salary')

### Plotting the Tree Decision

from sklearn import tree
plt.figure()
tree.plot_tree(DCR)
plt.show()

# =============================================================================
# Conclusion
# This program is a simple example to demonstrate how to apply the decision tree
# regression. Note that this model is not convenient if we have few  independent
# variables. In the results we can see a good prediction just in the original
# points, but for intermediate values, prediction is not satisfactory.
# =============================================================================


