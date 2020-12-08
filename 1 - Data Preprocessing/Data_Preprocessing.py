# =============================================================================
# Dataset preprocessing in Python for Machine Learning models
# This program is a simple exemple to domonstre how to take care about dataset 
# with missing data, non numerical variables and different scale. For this 
# example, we consider a dataset composed by 4 columns and 10 lines. The colums 
# are in order: Country, Age, Salary and Purchased.
# Library for Machine Learn: SciKitLearn.
# **Dataset Description**
# For this example, we consider a simple dataset composed by four columns and 
# ten rows. The dataset contains information about ten car purchase for a given 
# car store. The columns are: Country, Age, Salary and Purchased. The three 
# first columns are our features and the last is the response (dependent 
# variable) yes or not. Each row represents a custumer.
# =============================================================================

## Importing Libraries

import numpy as np
import matplotlib.pyplot as pĺt
import pandas as pd # One of the most library to read and treat a dataset

## Importing the Dataset

dataset = pd.read_csv('Data.csv') # reading and creating a dataset
X = dataset.iloc[:,:-1].values # Creating the independent variables (three first columns)
y = dataset.iloc[:,-1].values # Creating the dependent variable (response last column)

## Some dataset information

dataset.shape # To show the dataset format

dataset.head() # To read the first rows of dataset

dataset.describe() # To describe some statistical informations about numerical variables

dataset['Country'].value_counts().plot.bar() # To count and visualize the categorical variables

dataset['Purchased'].value_counts().plot.bar() # To count and visualize the categorical variables

dataset.isna().sum()

dataset.dtypes

# Note about the dataset #

# We visualized preliminary information about the dataset as some statistical for the numerical 
# variables and the plot bar to visualize the numerical variables. There are two important remarks 
# here, first we have missing data (one for salary and age), second we have also object variables, 
# it means, not numerical variable.

# One the most important thing to do when you work with a Machine Learning model is to avoid missing 
# data and convert object variables into numerical variables (here we must to transforms the follows 
#columns Country and Purchased).

## Taking care of missing data

# To take care of missing data we utilize one of must powerful Machine Learning library, Scikit-Learn 
# (this library has many modulus that contain several class).

# Here we are going to use SimpleImputer class. This class has an object to replace the nan values using 
# many strategies, such as mean and deviation (there are other options, you must to choose the better
# method according with your problem). To this case we are going to consider as strategt the mean value 
# to replace the nan value.


from sklearn.impute import SimpleImputer # calling the class and object
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # creating the object and passing the arguments 
imputer.fit(X[:, 1:3]) # fitting the columns Age and Salary
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replace the nan value into the mean value

print(X) # no more missing data

## Taking care of object variables ##

# Country Column #

# In our data we have the column Country which contains three countries as response. This impose to us a problem, 
# variables like tuple must be transformed in numbers. We have two way to do it. We transforms the variables into 
# 0, 1, 2 and so on or we transform into dummy variables, that's mean 0 or 1.

# For the country column we choose to use dummy variables, because it's more appropriate to linear regression models.
# In this case, the column country will be transformed into three columns, France, Germany and Spain. If France has 
# 1 as value it means that the other countries are 0.

# We utilize Compose class to transform our columns and Preprocessing class and the object OneHotEncoder 
# to convert the country's name into 0 or 1.

from sklearn.compose import ColumnTransformer # to transform the columns
from sklearn.preprocessing import OneHotEncoder # to convert object into number
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # creating the object and passing the arguments
X = np.array(ct.fit_transform(X)) # transforming the column

print(X) # no more country name

# Purchased Column #

# In this case our column Purchased is the response if a customer bought or not a SVU model. In the dataset these
#  responses are represented by yes or no, we need to change this categorical variable into a number. For this case, 
# the response will be transformed into 0 or 1 (not/yes).

# To transform our dependent variable we utilize Preprocessing class and the Label Encoder object.


print(y)

from sklearn.preprocessing import LabelEncoder # class and object to transform the dependent variable
le = LabelEncoder() # creating the object
y = le.fit_transform(y) # transfoming the variables

print(y)

## Splitting the Dataset into Training and Tese set

# In almost all models of Machine Learning, we split our dataset into two parts. Training set serves to train the model. 
# The Test set serves to compare our results with the real response.
# To split we use model_selection class and train_test_split object.

from sklearn.model_selection import train_test_split # class and object to split the variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1) # Creating the new sets and passing the arguments

# Test_size = 0.2 it means that 20% of our data will be separeted, in this case we will have two row and three columns to x_test and two row to y_test

print(X_train)

print(X_test)

## Feature Scaling

# In several real cases, some features can be in a different scale from the rest. To avoid overfitting and numerical error, 
# we transform this scale by the Standartization (this one method, but we have others). In this case the feature Salary shows 
# values in different scale from the rest, to solve it we utilize the class Preprocessin and the object StandardScale 
# (this a simple normalization into a guassian distribution)


print(X_train[:, 3])

from sklearn.preprocessing import StandardScaler # Class and object
sc = StandardScaler() # creating the object  
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) # fitting and transforming X_train
X_test[:, 3:] = sc.fit(X_test[:, 3:]) # fitting X_test

print(X_train[:, 3])

## Conclusion

# This code is a simple guide to data preprocessing using Scikit-Learn. Note, all these actions can be also made with Pandas 
# library and this example is an illustrative model. Examples with more complex feature engineering are presented on other
# folders from this repository.
