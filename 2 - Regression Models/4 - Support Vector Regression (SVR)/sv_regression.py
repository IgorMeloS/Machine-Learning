"""
Created on Fri Jun 26 15:02:30 2020
@author: igor
Machine Learning course - Support Vector Regression
"""

###Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#### Importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1 : -1].values
y = dataset.iloc[:, -1].values
#W need to reshape the array y because we have a row list and we need a column list
y = y.reshape(len(y),1)


#Here we need to proceed with feature scaling
#We utilize it because we have differents values for the independent and dependent variables
#In the first exemple that we used the linear regressor we did not make feature scaling
#Because there's no significative differences between the values and we have a linearity correlation
#To utilize SVR we need to proceed with the feature scaling in the both coordinates
#In our first exemple with feature scaling we did not make to y because we just had 1 or 0 now we have differents values

from sklearn.preprocessing import StandardScaler
#Here we need to make to variables objects for y and x
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

#Now we are going to make the SVR techinical, for it we utilize sklearn library
# We train whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Here we had trained the dataset, and we can have the prediction values for our model
#But, we need to be attentive because our values are on another scale due to feature scaling
#We can come back to the original scale using the reversal transform.

#Making one prediction for the level 6,5
#To make a prediction on the appropiate scale we must to apply the inverse_transform
#But attention, the inverse transform will be to y
# To the x variable we apply regressor.predict followed by sc_x.transform

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

#Visualizing the SVR results in feature scaling

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color='blue')

#Visualing the SVR results on the original scale

X_os = sc_x.inverse_transform(X)
y_os = sc_y.inverse_transform(y)

plt.scatter(X_os, y_os, color = 'red')
plt.plot(X_os, sc_y.inverse_transform(regressor.predict(X)), color='blue')

#For a smooth curve and higher precision

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid),1))
X_grid = sc_x.inverse_transform(X_grid)

plt.scatter(X_os, y_os, color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid))), color = 'blue')


