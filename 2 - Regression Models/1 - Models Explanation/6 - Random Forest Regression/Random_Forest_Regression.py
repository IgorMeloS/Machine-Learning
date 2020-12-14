# =============================================================================
# Random Forest Regression in Python using Scikit-Learn
# In this program we predict the salaries based on level position.
#
# **Dataset Description**
#
# The dataset to this model is composed by three columns and 1o row. We have
# one feature in the second column, Level Position. Our response is the last
# column, Salary. Based on the level, we construct a polynomial regression model
# to predict the salary for a given level position.
# =============================================================================

## Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv') # Reading and creating the dataframe
X = dataset.iloc[:, 1:-1].values # Independent Variable
y = dataset.iloc[:, -1].values # Dependent Variable

## Building the Model - Random Forest Regression

# The Random Forest Regression is an ensemble method.

# The goal of ensemble methods is to combine the predictions of several base estimators
# built with a given learning algorithm in order to improve generalizability / robustness
# over a single estimator.
# This method consists in many repetitions of decision tree regression. The trees
# are chose randomly and we consider the total average to fit the model.
# In Sckit-Learn we consider the classa Ensemble and the object Random Forest Regressor.

from sklearn.ensemble import RandomForestRegressor # Class and object
RFR = RandomForestRegressor(n_estimators=10, random_state=0) # Creating the object
# n_estimatimators is the number of decsion tree that we utilize to train our model
RFR.fit(X,y) # Fitting and training the model

### Making a Single Prediction

RFR.predict([[6.5]])

### Visualizing the Results

X_grid = np.arange(min(X), max(X), 0.001) # Increasing the number of point
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red') # Scatter Plot
plt.plot(X_grid, RFR.predict(X_grid), color='blue') # Plotting the Random Forest Curve
plt.title('Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# =============================================================================
# Conclusion
# This example is a simple demonstration how to apply the random forest
# regression. This method is very similar to Decision Tree Regression. But here
# we have a certain number of decision tree chose randomly and from these
# repetitions we take the mean. The results is satisfactory and we have more
# information about intermediate values, these values do not appear in a single
# decision tree model.
# =============================================================================

