"""
Created on Thu Jun 25 13:34:57 2020
@author: igor
Machine Learning Course - Polynomial Regression model
"""

#####Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#####Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
#Here we do not have interest in the first column, for this raison, we ignore it
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Here, we train whole dataset in linear regression model
#We make the linear regression like a comparison to the polynomial model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Once time that you trained the original dataset in linear regression
#We make the same for the polynomial regression
#Here we use sklearn preprocessing with the class PolynomialRegression
from sklearn.preprocessing import PolynomialFeatures
#In the function PolynomialFeature we must to choose the degree of our function
#You can choose many degrees and visualize what is the better for your model
poly_reg = PolynomialFeatures(degree=6)
#After we must to declare a new variable to make the polynomial regression, remember we make for the whole data
X_poly = poly_reg.fit_transform(X)
#We declare a new variable to make another linear regression
lin_reg_2 = LinearRegression()
#We fit in linear regression our matrix feature with the dependent variable
lin_reg_2.fit(X_poly, y)

#Visualizing the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.show()
#How we can observate, this is not a good fit to the curve, for this raison we need to the polynomial regression

#Visualizing the polynomial regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.show()

#Visualizing the polynomial regression (higher resolution and smoother curve)
X_poly_grid = np.arange(min(X), max(X),0.01)
X_poly_grid = X_poly_grid.reshape((len(X_poly_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_poly_grid, lin_reg_2.predict(poly_reg.fit_transform(X_poly_grid)), color='blue')





