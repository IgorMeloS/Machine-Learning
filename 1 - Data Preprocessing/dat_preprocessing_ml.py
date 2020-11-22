# =============================================================================
# Dataset preprocessing in Python for Machine Learning models
# Program wrote by Igor Melo.
# August, 2020.
# This program is a simple exemple to domonstre how to take care about dataset,
# we consider a dataset composed by 4 columns and 10 lines. The colums are in
# order: Country, Age, Salary and Purchased.
# Library for Machine Learn: SciKitLearn.
# =============================================================================

# Libraries (the first step, import the libraries to read and take care the datas)
# Three basics libraries, NumPy, Matplotlib and Pandas (to read csv file)

import numpy as np
import matplotlib.pyplot as pĺt
import pandas as pd

# Importing the dataset using pandas.

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# In this dataset our interest is to make a model to predict a sale of SUV model.
# We consider two variables, independent and dependent variables, in this case
# y is the dependent variable and contains the datas by the column Purchased.
# The three first columns we consider like independent variables.

# Missing data (In the columns 1 and 2 we have the nan values, we need to change it)
# If your dataset presents this problem, you can use sklearn.impute object and
# SimpleImputer class. This class has a function to replace the nan values
# using many strategies, as mean and deviantion.

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# In our data we have the column Country. This impose to us a problem, variables
# like tuple must be transformed in numbers, to do it, we utilize the categorical
# data. Objects to use: sklearn.compose (class: ColumnTransform) and
# sklearn.preprocessing (class: OneHotEncoder)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Now we have the countries in three differents columns, each column represents
# a country with the value 1 otherwise 0.

# In our case, the dependent variable is represented by 'yes' or 'no'.  We change
# it using a transformation to change this values in a binary combination 1 or 0.
# Here we call the object sklearn.preproprecessing (class: LabelEnconder).

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Training and Test set
# Here we prepare our dataset to a machine learn model. We split our dataset
# into two parts, training and test set. The first is more big and the second
# is used to make comparation between the reals values and the predict values gave
# by the machine learn model. To split we use the object sklearn.model selection
# (class: train_test_split).

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1)

# The last action to finish the data preprocessing is the transformation of some
# inpendent values. In fact, we make a Standalization to have the same dependent
# variable magnitude. Object sklearn.preprocessing (class: StandardScaler)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit(X_test[:, 3:])

# =============================================================================
# This code is a simple exemple and serve as basic guide to data preprocessing
# using SciKitLearn. Is possible possible that we do not use all objects presented
# in every model. But the basic to start a good model is here. We can explore more
# about each class and yours functions, but here is a simple introduction. Each
# case demands its specificities, with this objects we can explore it.
# From here, you can code your manchine learn model. Let's do it.
# =============================================================================
print('test')
