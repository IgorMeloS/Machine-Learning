# =============================================================================
# Simple Linear Regression in Python using ScikitLearn
# Program wrote by Igor Melo,.
# August, 2020.
# This program is a simple exemple to domonstre a linear regression model
# Here we want to predict the salary based in years of experience
# =============================================================================

# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# This dataset is composed by two columns, years of experience and salary.
# The independent variable is year of experience and we want to construct a model
# to predict the salary, the dependent variable.

# Spliting the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# Observation, in the function train_test_split the argument test_size define the percentual
# of data dedicated to the y_test variable. In our case 20%.

# Now, we construct our model. The linear function is Y = a + b*X
# Based on the datas the model will try to find the ideal curve, using a squared
# difference model, to give a optimal b coeficient. From this moment we can predict
# the salary based on the years of experience. We can also compare our results with
# the real datas to verify the accuracy.

# Simple linear Regression Model - skelearn.linear_model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Here we fit the datas to obtain the better curve, we train X_train and y_train
regressor.fit(X_train, y_train)
# Now the regression function knows the new curve and we can predict new salaries
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)
# To make a simple comparation we print the y_pred and y_test variables
# You can also see a single value, to do it we put just the years of exeperience.
print(y_pred_test)
print(y_test)

# To know the coeficients a and b
print(regressor.coef_)
print(regressor.intercept_)

# In this case the best-fitting is gave by Y = 25609.89799835482 + 9332.94473799*X

# Visualizing the prediction curve and reals datas

# Train set #
plt.title('Salary vs Experience (Training set)')
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_pred_train, color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# Test set #
plt.title('Salary vs Experience (Test set)')
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred_test, color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel.('Salary')

# =============================================================================
# This program was a simple demonstration of a simple linear regression model.
# We can employ it in others pratical exemple and make a good predictions when
# you have a linear problem. It can be very usefful in a take decesion precess,
# profit optimization and other exemples in the real world.
# =============================================================================
