# =============================================================================
# Multiple Linear Regression in Python using SciKitLearn
# Program wrote by Igor Melo 
# August, 2020
# In this program we want to predict startups profits using a dataset composed
# by five columns
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Our dataset is composed by five columns, four colums to describe the independent
# variables (spending and cities) and the last one to dependent variable, in this case, startups profits
# R&D Sepend, Administration, Marketing Spend, City and Profit.

# Due to have a columns like city, we must to transform the tuple in number values
# Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Now, the cities are represented by number 0 or 1 in three columns

# Splitting the dataset into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# From this moment, we can construct our multiple linear regression model
# We have many model, as Backward, Forward, All in and many others. Which one should I choose.
# The method more directly is Backward eleminination, but the ScikitLearn library takes care
# and chooses the better methods, and construct a good model by variable elimination.
# To do it, we utlize the same object that we have used to the simple linear regression

# Contructing the multiple linear regression model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Now we have a trained model and we are able to make prediction for this dataset

# Making predictions

y_pred_test = regressor.predict(X_test)


# Visualizing the predicteds profits
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_test.reshape(len(y_pred_test),1), y_test.reshape(len(y_test),1)), 1))

# =============================================================================
# In this simple exemple, we do not visualize any graph in reason to the multiple
# dimensions, we can visualize in three dimensions, but here, this not the objective.
# The objective of this program was to contruct a multiple linear regression
# model to predict the startups profits using a dataset with several differents
# information. Our results are convicing compared to reals datas.
# =============================================================================
