# =============================================================================
# Polynomial Regression in Python using ScikitLear
# Program wrote by Igor Melo
# August, 2020
# In this program we predict the salaries based on level position. Following
# the dataset we can see that the relation between salary and position is not
# linear, for this reason we implement a polynomial linear regression
# =============================================================================

# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv')
# In this dataset we can see tree columns. The first column is completely despicable
# because we just have the names of position. So, we are interested in two last columns,
# level position and salary. To construct a good model we delete the first column.
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# One important think to know is, the polynomial regression is a linear regression
# because the polynomial correlation resides in the independent variables but
# the coefficients is kept linear, so this function is polynomial,
# but the method is linear. For this reason, the first step is simple linear regression
# to obtain a good coefficient

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Here we have the best-fit coefficient, now we must to construct the polynomial
# relation with the independent variable, to do it, we use sklearn.preprocessing

# Once time we have the linear coefficient we transform our feature matrix to put
# the values in a polynomial relation, it's meas we transform the values of X in 
# squared values to obtain an equation like y = b + b1*x + b2*x^2+...+bnx^n

from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Here we have the model to predict values using the polynomial regression
# The prediction is made using a linear prediction with the squared values obtained in the last step

# We can visualize our results with some graphs due to have just two variables

# Visualizing the linear results

plt.title('Truth or Bluff (Polynomial Regression)')
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
# We use this graph to vizualize that the linear aproximation is not a good fit

# Ploting the polynomial linear regression

plt.title('Truth or Bluff (Polynomial Regression)')
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(X_poly), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
# Here we see the best-fit curve in comparison with the datas (red points)

# Plotting the polynomial with more precision points (better plot)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.title('Truth or Bluff (Polynomial Regression)')
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
# Here we have a more smoother curver

# To change the curve and try to obtain a better fit, you can change the degree 
# in the polynomial object poly_reg

print(lin_reg2.coef_)
print(lin_reg2.intercept_)

# To make a simple prediction based on level position
print(lin_reg2.predict(poly_reg.fit_transform([[10]])))

# =============================================================================
# This code is a good explanation about polynomial regression. 
# We make a linear regression to describe the best-fit curve to a polynomial model
# it's possible because the coefficients of the polynomial equation are linear and
# the polynomial relation is presented just in the independents variable.
# So, we fit the best coefficients and we transforma the feature matrix according
# with the polynomial equation. Here is a simple, but robuste exemple that can be
# employed to other reals problems.
# =============================================================================
