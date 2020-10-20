#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:11:42 2020
@author: igor
Machine Learning course - Multiple Linear Regression
"""

#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values

#Categorical data transform to the dummy variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Spliting Dataset into to Training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
Today the objective is to construct a multple linear regression model
Thre're several methods to make a multiple linear regression
So, in python when you use scikitlearn we do not must to choose the method to employ
Python makes it for us.
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#In fact, when you work with multiple linear regression, there's a difficult to plot
#the original dataset because we have many variables, for this raison, normally
#we compare the y_test with your prediction to compare the accurancy of our method
#But, keep in mind, some time we need to plot the our results
#in this case we must to use some method as backward elimination.

####Creating the prediction to test set

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
### Here we compare the y_pred with y_test we're going to concatenate the vecors to have a better comparison 
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#plt.scatter(X_train[:,3], y_train, color = 'red')
#plt.plot(X_train[:,3], regressor.predict(X_train[:,3]), color = 'blue')
#plt.show()
##We separate the test set to verify if this values have a good accurancy with
## the prediction made by the traing set
##Now we're going to visualize the test set results
##It's a good observation, the values of y_pred corresponds to the values of the training set prediction
#plt.scatter(X_test, y_test, color = 'red')
#plt.scatter(X_test, y_pred, color = 'black')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.show()