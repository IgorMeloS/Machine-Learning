# =============================================================================
# Simple Linear Regression in Python using ScikitLearn
# This program is a simple example to demonstrate a linear regression model. 
# Here we want to predict the salary based on years of experience.
#
# **Dataset Description**
#
# The dataset to this model is composed by two columns and 31 row. We have one 
# feature in the first column, Experience. Our response is the last column, 
# Salary. Based on the experience, we construct a linear regression model to 
# predict the salary for a given years of experience.
# =============================================================================

## Importing Libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the Dateset

dataset = pd.read_csv('Salary_Data.csv') # Creating and reading the dataset
X = dataset.iloc[:, :-1].values # Independent Variable
y = dataset.iloc[:, 1].values # Dependent Variable

## Splitting the Dataset into Training and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

## Constructing the model - Simple Linear Regression

# Now, we construct our model. A linear function is defined as $Y(X) = a + b*X$.
# Based on the datas the model will try to find the ideal curve, using a squared 
# difference model $\min_{b}||Xb - y||^{2}_{2}$, to give a optimal $b$ coefficient.
# From this moment, we can predict the salary based on the years of experience.
# We can also compare our results with the real data to verify the accuracy.
# We utilize Linear_Model class and LinearRegression object from Scikit-Learn.


from sklearn.linear_model import LinearRegression # Class and object
regressor = LinearRegression() # Creating the object
regressor.fit(X_train, y_train) # Training the model, X_train and y_train

### Predicting New Results

# Once time our model is trained, we can predict new values. To do it, we utilize the attribute .predict


y_pred = regressor.predict(X_test) # Creating new variable to allocate the predicted results
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1)) # To compare predict and real results

# To know the coefficient b of the linear function
print(regressor.coef_)
print(regressor.intercept_)

# The linear model constucted by the linear regression is represented by $Y(X) = 25609.89799835482 + 9332.94473799*X$, this equation is the predicted curve.

### Visualizing the predicted curve with the real model

#### Training Set


plt.figure() # Creating the figure
plt.scatter(X_train, y_train, color = 'red') #Scatter Plot for the training set, the points distributed 
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Plotting the predict curve
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#### Test Set

plt.figure() # Creating the figure
plt.scatter(X_test, y_test, color = 'red') #Scatter Plot for the test set, the points distributed 
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Plotting the predict curve
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

## Conclusion

# This program is a little demonstration of a simple linear regression model. 
# The results are satisfactory with good predictions. We can employ it in others 
# practical example and make a good predictions when you have a linear problem. 
# It can be very useful in a take decision process, profit optimization and other 
# examples in the real world.
