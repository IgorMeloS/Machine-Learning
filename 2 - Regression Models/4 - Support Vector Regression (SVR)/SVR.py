# =============================================================================
# Support Vector Regression (SVR) in Python using ScikitLearn
# Program wrote by Igor Melo.
# In this program we predict the salaries based on level position. Following
# the dataset we can see that the relation between salary and position is not
# linear, for this reason we implement a SVR to predict.
# =============================================================================

# Importing Libraries

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

# Feature Scaling
# In this model we need to aply feature scaling, for one simple reason,
# the SVR model is not a linear combination between  X and y.
# Thre is an other trick behind the method. Then we need to have all
# independent variables in the same scale to make a good prediction.
# Note that in all situation that we have a linear combination between
# X and y we do not need to apply feature scaling.

# Before proceeding with the feature scaling we must change the array y into a 2D
# array. We must do it because the feature scaling object just transforms 2D arrays

y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model
# Due to the correlation between X and y in the dataset, we do not need to split
# the dataset into train and test. So here, we are going to train the model on
# the whole dataset. We call a new class in scikitlearn libraary, sklearn.svm
# and we utilize the object SVR.
# In the object SVR we have many arguments, that are model of distributions.
# The better distribution model for this method is Hyperbolic Tangent Kernel.
# But all dataset is a linear combination and follow a normal distribution.
# For this reason, we utilize Gaussian Radial Basis Function (RBF)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf', degree=6)
regressor.fit(X,y)

# Now we can make some predictions, our model is trained on whole dataset.
# Heads-up!!! We can not make a directly prediction we employed a feature scaling.
# Before any prediction, we need come back to the original scale.

# A single prediction.
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[10]]))))
# Like the salary is the dependent variable to be predict, in this single prediction
# we used the inverse transformation. But be careful, to visualize the results
# you must aplly the the inverse transformation to the all variables.

# Visualizing the results
X_os = sc_X.inverse_transform(X)
y_os = sc_y.inverse_transform(y)

plt.title('Truth or Bluff (Support Vector Regression)')
plt.scatter(X_os, y_os, color = 'red')
# plotting the model with predictions and your transformation
plt.plot(X_os, sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Visualising the SVR results (for higher resolution and smoother curve)

X_grid = np.arange(min(X_os), max(X_os), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid_os = sc_X.transform(X_grid)

plt.scatter(X_os,y_os, color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(X_grid_os)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# =============================================================================
# This program is a simple exemple to demonstre how to use a SVR model.
# The model is different to the linear regression and the theory behind the method
# is very interesting and aid to understand better. I suggest a quickly read of it.
# You most also try to fit a best model using the arguments in the object SVR().
# Enjoy machine learn.
# =============================================================================
