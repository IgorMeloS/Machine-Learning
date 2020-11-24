# -*- coding: utf-8 -*-
"""Breast_Cancer_Igor.ipynb

Automatical ly generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vh5cQVmFhvweQoUWA9D4Myz8OkJ2zB4R

# Breast Cancer Classification
This program has....

# Importing Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""# Importing Dataset"""

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()

type(cancer_data) # Cancer_data is a Bunch, it's means a Pyton Dictionary.

print(cancer_data)

# We need to know all keys inside the dictionary
cancer_data.keys()
# We have six keys. Data that contains the all features, target the variable who we want to predict, target name, DESCR is the description of problem, feature names and the filename.

# It's important to know problem description
print(cancer_data['DESCR'])

# As we can see in the description the target names are Malignant or Benign. Here we visualize these values
print(cancer_data['target'])
# So we associate 0 to Malignant and 1 to Benign

# It's important to know the feature names
print(cancer_data['feature_names'])

# We can also visualize the values of features
print(cancer_data['data'])

"""# Creating the Dataframe

"""

cancer_df = pd.DataFrame(np.c_[cancer_data['data'], cancer_data['target']], columns=np.append(cancer_data['feature_names'],['target']))
# Here we employed np.c_ to joint the feature with the target, to assign the columns names we utilize np.append and we passed the respectives keys

"""## Visualizing the head, tail and some statistical informations about the dataframe"""

cancer_df.head(10)

cancer_df.tail(10)

cancer_df.shape

cancer_df.describe()

cancer_df['target'].value_counts()

cancer_df['target'].value_counts().plot.bar()

"""# Visualizing some data informations (seaborn)

## Correlation
The correlation is a important step to understand the relation between the feature values, it implicates how much a certain variable depends on the other variable.
"""

# Correlation between all feature values
plt.figure(figsize=(20,10))
sns.heatmap(cancer_df.corr(), annot=True)

# Now, we want to visualize highly correlated variables, we choose 0.9 because this value has a great significance
corr = cancer_df.corr()
kot = corr[corr>=.9]
plt.figure(figsize=(20,10))
sns.heatmap(kot, cmap="Greens", annot=True)





"""## Pairplot (Distribution and scatter plot)
Here we select the variable that has high correletion with one other, we can see these variable in the graphic above.
"""

sns.pairplot(cancer_df, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean concavity', 'mean concave points', 'radius error','perimeter error', 'area error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst concave points'] )

"""From the graphic we can conclude that all these features present different distribution beteen the the target variable, it means that each feature preseted here has an importance to model and must be considered to train the model of selection prediction.

## Scatter plot
Here, we visualize the scatter plot for the features that have correlation 1 and 0.99.
We have these graphics above, but here is more zoomed.
"""

sns.scatterplot(x = 'mean perimeter', y = 'mean radius', hue = 'target', data = cancer_df)

sns.scatterplot(x ='mean area' , y = 'mean perimeter', hue = 'target', data = cancer_df)

"""Maybe I'll put an explanation here to describe better the statistical feature.

# Model Setting
Here we define the independent and dependent variables, training and test set. We apply feature scaling.

## Independent and Dependent Variables
"""

X = cancer_df.drop(['target'], axis = 1)
y = cancer_df['target']

"""## Splitting the Dataset into Training and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""## Feature scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

"""# Training and Selection Model

# Training the model
"""

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

# K Nearest Nieghbors

from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p = 2)
classifier2.fit(X_train, y_train)

# Support Vector Machine - Linear Classifier

from sklearn.svm import SVC
classifier3 = SVC(kernel= 'linear', random_state=0)
classifier3.fit(X_train, y_train)

# Kernel Support Vector Machine

from sklearn.svm import SVC
classifier4 = SVC(kernel='rbf', random_state=0)
classifier4.fit(X_train, y_train)

# Naïves Bayes Classification

from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)

# Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier6.fit(X_train, y_train)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
classifier7.fit(X_train, y_train)

"""## Predicting the test result"""

y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)
y_pred4 = classifier4.predict(X_test)
y_pred5 = classifier5.predict(X_test)
y_pred6 = classifier5.predict(X_test)
y_pred7 = classifier7.predict(X_test)

"""## Selection Model"""

from sklearn.metrics import classification_report, confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
cm4 = confusion_matrix(y_test, y_pred4)Why did you not apply feature scaling directly instead of to write the normalization ? I think that it's more simple, fast and it gives good results.
cm5 = confusion_matrix(y_test, y_pred5)
cm6 = confusion_matrix(y_test, y_pred6)
cm7 = confusion_matrix(y_test, y_pred7)
print('Checking Confusion Matrix to a single observation')
print('Logistic Regression Classification')
print(cm1)
print(classification_report(y_test,y_pred1))
print('\n')
print('K Nearest Neighbors')
print(cm2)
print(classification_report(y_test,y_pred2))
print('\n')
print('Support Vector Machine')
print(cm3)
print(classification_report(y_test,y_pred3))
print('\n')
print('Kernel Support Vector Machine')
print(cm4)
print(classification_report(y_test,y_pred4))
print('\n')
print('Naive Bayes Classification')
print(cm5)
print(classification_report(y_test,y_pred5))
print('\n')
print('Decision Tree Classification')
print(cm6)
print(classification_report(y_test,y_pred6))
print('\n')
print('Random Forest Classification')
print(cm7)
print(classification_report(y_test,y_pred7))
print('\n')

from sklearn.model_selection import cross_val_score
accuracies1 = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
accuracies2 = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 10)
accuracies3 = cross_val_score(estimator = classifier3, X = X_train, y = y_train, cv = 10)
accuracies4 = cross_val_score(estimator = classifier4, X = X_train, y = y_train, cv = 10)
accuracies5 = cross_val_score(estimator = classifier5, X = X_train, y = y_train, cv = 10)
accuracies6 = cross_val_score(estimator = classifier6, X = X_train, y = y_train, cv = 10)
accuracies7 = cross_val_score(estimator = classifier7, X = X_train, y = y_train, cv = 10)

print('Checking K-Fold Cross Validation')
print('\n')
print('Logistic Regression Classification')
print("Accuracy: {:.2f} %".format(accuracies1.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies1.std()*100))
print('\n')
print('K Nearest Neighbors')
print("Accuracy: {:.2f} %".format(accuracies2.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies2.std()*100))
print('\n')
print('Support Vector Machine')
print("Accuracy: {:.2f} %".format(accuracies3.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies3.std()*100))
print('\n')
print('Kernel Support Vector Machine')
print("Accuracy: {:.2f} %".format(accuracies4.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies4.std()*100))
print('\n')
print('Naive Bayes Classification')
print("Accuracy: {:.2f} %".format(accuracies5.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies5.std()*100))
print('\n')
print('Decision Tree Classification')
print("Accuracy: {:.2f} %".format(accuracies6.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies6.std()*100))
print('\n')
print('Random Forest Classification')
print("Accuracy1: {:.2f} %".format(accuracies7.mean()*100))
print("Standard1 Deviation: {:.2f} %".format(accuracies7.std()*100))

"""Here we select the three best score, we consider the accuracy score and we analize the Standard Deviation, results with high Deviation, can descondered, it depend on the accuracy score.
For this model we select


1.   Logistic Regression
2.   Support Vector Machine
3.   Kernel Support Vector Machine

# Boosting the model
"""

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train,y_train)

grid.best_params_

grid.best_estimator_

grid_predictions = grid.predict(X_test)

cm = confusion_matrix(y_test, grid_predictions)

sns.heatmap(cm, annot=True)

print(classification_report(y_test,grid_predictions))

classifier4b = SVC(C = 10, kernel='rbf', gamma = 0.01, random_state=0)
classifier4b.fit(X_train, y_train)
y_pred4b = classifier4b.predict(X_test)

accuracies4b = cross_val_score(estimator = classifier4b, X = X_train, y = y_train, cv = 10)

print('Kernel Support Vector Machine')
print("Accuracy: {:.2f} %".format(accuracies4b.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies4b.std()*100))
