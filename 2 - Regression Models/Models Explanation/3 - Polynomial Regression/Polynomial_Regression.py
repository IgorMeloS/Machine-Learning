# =============================================================================
# Polynomial Regression in Python using ScikitLear
# This program is a simple example to demonstrate a polynomial regression model. 
# Here we want to predict the salary based on positon level.
#
# **Dataset Description**
#
# The dataset to this model is composed by three columns and 1o row. We have one  
# feature in the second column, Level Position. Our response is the last column,
#  Salary. Based on the level, we construct a polynomial regression model to 
# predict the salary for a given level position.
# =============================================================================

## Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the Dataset

dataset = pd.read_csv('Position_Salaries.csv') # Creating and reading the dataset
X = dataset.iloc[:, 1:-1].values # Independent Variable
y = dataset.iloc[:, -1].values # Dependent Variable

dataset.head()

## Constructing the model - Polynomial Regression

# One important think to know is, the polynomial regression is a linear regression
# because the polynomial correlation resides on the independent variables, but
# the coefficients is kept linear. So in the polynomial regressions the method is linear. 
# A polynomial function can be represented by $Y = a + b_{1}*X + b_{2}*X^{2}+...+b_{n}*X^{n}$.
# The method to make a polynomial regression contains two step. First, we must to
# know the coefficients $b_{i}$ using a linear regression function. Second fit the
# independent variables into a polynomial degrees.

### Fitting the coefficients

from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression() # Creating the linear regression object
lin_reg.fit(X, y) # Fitting the coefficients

### Polynomial Features

from sklearn.preprocessing import PolynomialFeatures # Class and object
poly_reg = PolynomialFeatures(degree = 6) # Creating the object to convert the features into polynomial features
# the argument degree can be changed to try the best fit.
X_poly = poly_reg.fit_transform(X) # Creating the polynomial variable
lin_reg_2 = LinearRegression() # Creating the object to fit the polynomial features
lin_reg_2.fit(X_poly, y) # Fitting the polynomial features

### The coefficients

print(lin_reg_2.coef_)
print(lin_reg_2.intercept_)

## Visualizing the Results

### Linear Results

# Here we can visualize that linear regression model isn't the best fit current problem.

plt.scatter(X, y, color = 'red') # Scatter plotting
plt.plot(X, lin_reg.predict(X), color = 'blue') # Linear Regression Plot
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

### Polynomial Results

X_grid = np.arange(min(X), max(X), 0.1) # Increasing the number of point to obtain a smoother curve 
X_grid = X_grid.reshape((len(X_grid), 1)) # Reshaping
plt.scatter(X, y, color = 'red') # Scatter plot
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') # Plotting the polynomial curve
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# =============================================================================
# Conclusion
# This code is a good explanation about polynomial regression. We make a linear
# regression to describe the best-fit coefficients and then we apply the polynomial
# features to obtain our predicted curve. The model shows a good response.
# =============================================================================