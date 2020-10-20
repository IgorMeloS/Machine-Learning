# =============================================================================
# Random Forest Regression in Python using Scikit-Learn
# Program wrote by Igor Melo, September 20.
# This program wants to predict a new salary training a random forest regression
# model. The data set is composed by 3 columns: Name Position, Level Position and
# Salary. We just consider the two last.
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Random Forest Regression model 

# The random forest regression, is a similar model to the decision tree model. But
# here, the difference consists in make a sample of decision tree regression randomly 
# chosen.After it, we take the mean to obtain the final result. In summary, this is 
# a ensemble model. Due to make more measures, this method is more efficient in 
# some cases, with a better accuracy in comparison to decision regression tree.
    
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=10, random_state=0)
# n_estimatimators is the number of tree that we utilize to train our model
RFR.fit(X,y)

# Making a simple prediction

RFR.predict([[6.5]])

# Visualizing the prediction results (Higher Resolution)

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, RFR.predict(X_grid), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# =============================================================================
# This exemple is a simple demonstration how to apply the random forest regression.
# This method is very similar to Decision Tree Regression. But here we havy more
# tree randomly chosed. For this reason, we can visualize in the previsions that
# between the points we have to steps. This fact makes forest random regression
# more aaccurate.
# =============================================================================
