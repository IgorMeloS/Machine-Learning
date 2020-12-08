# =============================================================================
# Decision Tree Regression in Python using Scikit-Learn
# Program wrote by Igor Melo.
# In this program we predict the salaries based on level position. Following
# the dataset we can see that the relation between salary and position is not
# linear. We propose a decision tree method to predictic new salaries
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
# We did not consider the first collumn, Name position.
y = dataset.iloc[:, -1].values

# Training Decision Tree Regression model

# Decision tree regression is a model that uses the tree classification, devising 
#the data set in branches  . Once time we have the firsts branches we qualify it 
# in true or false. Then we can define new branches and take statistical measures
# like  mean and deviation. From those results we construct our regression and 
# classifier model.  

from sklearn.tree import DecisionTreeRegressor
DCR = DecisionTreeRegressor(random_state=0)
DCR.fit(X,y)
# Now we have the trained model and we can make some predictions

# Making a new prediction with the regressor model

print(DCR.predict([[7]]))

# Visualizing the prediction model (Higher Resolution)
# Here we consider a larger numbers of points to obtain a smooth curve.
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, DCR.predict(X_grid))
plt.title('Decision Tree Regression')
plt.xlabel('Level Position')
plt.ylabel('Salary')

# =============================================================================
# This program is a simple exemple to domonstrate how to apply the decision tree
# regression. Note that this model is not convenient if we have few 
# independent variables. 
# In the results we can see a good prediction just in the original points, but
# for intermediate values, prediction is not satisfactory.
# =============================================================================
