# =============================================================================
# Multiple Linear Regression in Python using Scikit-Learn
# In this program we want to predict startups profits using a linear regression 
# model.
#
# **Dataset Description**
#
# The dataset is composed by 50 rows that correspond the total of startup and 5 
# columns. The features of the problem are: R&D Spend, Administration, Marketing 
# Spend and City. The response is the Profit column. Based on these feature we 
# are going to construct the model.
# =============================================================================


## Importing Libraries

import numpy as np
import pandas as pd

## Importing the Dataset

dataset = pd.read_csv('50_Startups.csv') # Reading the dataset and creating the dataframe
X = dataset.iloc[:, :-1].values # Creating feature array
y = dataset.iloc[:, -1].values # Creating the response array

## Data Preprocessing

# In this problem the column City must be modified into numerical values.


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X[0:5])

## Splitting the Dataset into Training and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

## Constructing the model - Multiple Linear Regression

# A Multiple Linear Regression is very similar to the Linear Regression, the difference
# consists on the number of independent variables and their coefficients.
# The linear equation is $Y = a + b_{1}X_{1}+..+b_{n}X_{n}$. There are several methods
# to solve this kind of problem (Backward and Forward elimination, for example).
# The Scikit-Learn library uses the Ordinary Least Square.
# We utilize the same class and object for a linear regression model.


from sklearn.linear_model import LinearRegression # Class and object
regressor = LinearRegression() # Creating the object
regressor.fit(X_train, y_train) # Fitting the model

### Predicting  New Results

y_pred_test = regressor.predict(X_test)

# **Note**: In this problem we can't have a graphic visualization due to have more than two variables.

### Visualizing the Predicted Results


np.set_printoptions(precision=2)
print(np.concatenate((y_pred_test.reshape(len(y_pred_test),1), y_test.reshape(len(y_test),1)), 1))

# =============================================================================
# Conclusion
# This program was a demonstration how to build a multiple linear regression 
# model. The predicted results are near to reals results. Here We do not discuss 
# the metrics questions. It'll be showed in the futures cases.
# =============================================================================