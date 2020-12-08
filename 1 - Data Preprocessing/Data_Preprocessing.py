#!/usr/bin/env python
# coding: utf-8

# #  Dataset preprocessing in Python for Machine Learning models
# 
# This program is a simple exemple to domonstre how to take care about dataset with missing data, non numerical variables and different scale. For this example, we consider a dataset composed by 4 columns and 10 lines. The colums are in order: Country, Age, Salary and Purchased.
# Library for Machine Learn: SciKitLearn.
# 
# **Dataset Description**
# 
# For this example, we consider a simple dataset composed by four columns and ten rows. The dataset contains information about ten car purchase for a given car store. The columns are: Country, Age, Salary and Purchased. The three first columns are our features and the last is the response (dependent variable) yes or not. Each row represents a custumer.

# ## Importing Libraries 
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as pĺt
import pandas as pd # One of the most library to read and treat a dataset


# ## Importing the Dataset

# In[2]:


dataset = pd.read_csv('Data.csv') # reading and creating a dataset
X = dataset.iloc[:,:-1].values # Creating the independent variables (three first columns)
y = dataset.iloc[:,-1].values # Creating the dependent variable (response last column)


# ## Some dataset information

# In[5]:


dataset.shape # To show the dataset format


# In[3]:


dataset.head() # To read the first rows of dataset


# In[6]:


dataset.describe() # To describe some statistical informations about numerical variables


# In[13]:


dataset['Country'].value_counts().plot.bar() # To count and visualize the categorical variables


# In[19]:


dataset['Purchased'].value_counts().plot.bar() # To count and visualize the categorical variables


# In[28]:


dataset.isna().sum()


# In[27]:


dataset.dtypes


# **Note about the dataset**
# 
# We visualized preliminary information about the dataset as some statistical for the numerical variables and the plot bar to visualize the numerical variables. There are two important remarks here, first we have missing data (one for salary and age), second we have also object variables, it means, not numerical variable.
# 
# One the most important thing to do when you work with a Machine Learning model is to avoid missing data and convert object variables into numerical variables (here we must to transforms the follows columns Country and Purchased). 

# ## Taking care of missing data
# 
# To take care of missing data we utilize one of must powerful Machine Learning library, Scikit-Learn (this library has many modulus that contain several class).
# Here we are going to use SimpleImputer class. This class has an object to replace the nan values using many strategies, such as mean and deviation (there are other option, you must to choose the better method according with your problem).

# In[ ]:




