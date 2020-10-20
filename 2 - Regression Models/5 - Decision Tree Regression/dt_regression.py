"""
Created on Fri Jun 26 17:23:11 2020
@author: igor

"""

###Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###Importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1 : -1].values
y = dataset.iloc[:, -1].values

#Training the dataset with decision tree method. We train whole dataset
#To make it we utilize sklearn with the class tree and the object DecisionTree

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

#Predicting one single value
regressor.predict([[6.5]])

#Visualizing the tree decision results

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')

#Higher precision and smooth curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
print(X_grid)

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
