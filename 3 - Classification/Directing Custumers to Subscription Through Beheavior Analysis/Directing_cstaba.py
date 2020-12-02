# -*- coding: utf-8 -*-
"""Directing_cstaba.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oZzta1tdNP1T10QsZaQdELVAJ1GqCAdL

# Directing Customers to Subscription Through App Behavior Analysis


In marketing analysis, the Machine Learning technics can provide a helpful tool. For example, companies that have mobile services in two version free and paid, want always to obtain the maximum number of subscription. One way to make the better publicities is to know the customers behavior, based on the behavior of each custumer some company can offer your mobile services.
But the question is: How can we know the custumers behavior ?
The answer for this question is simple, we construct a model of classification based on the client acitivities in the free services and then, we can classify if a certain custumer will subscripe or not.

# Data Preprocessing

## Importing Libraries
"""

import pandas as pd
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""## Importing Dataset"""

dataset = pd.read_csv('appdata10.csv')
dataset.shape

dataset.dtypes

dataset.head()

dataset['hour'] = dataset.hour.str.slice(1,3).astype(int) # the values in the column hours are object variable we need to change it

dataset.describe()

"""## Analising numerical variables"""

dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'first_open', 'enrolled', 'enrolled_date'])

dataset2.head()

"""### Histogram"""

plt.figure(figsize=(20,10))
plt.suptitle('Histograms of numerical variables', fontsize = 20,)
for i in range(1, dataset2.shape[1] + 1):

  plt.subplot(3, 3, i)
  f = plt.gca()
  f.set_title(dataset2.columns.values[i-1])
  vals = np.size(dataset2.iloc[:, i-1].unique())
  plt.hist(dataset2.iloc[:, i-1], bins= vals, color = '#3F5D7D' )

"""### Correlation Plot"""

dataset.corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
                                            title = 'Correlation with reponse variable',
                                            fontsize = 15, rot = 45, grid = True, color = '#5F5D7D')

"""### Correlation Matrix"""

sns.set(style='white', font_scale= 1)
corr = dataset2.corr() # here we compute the correlation between numericals variables
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype= np.bool) # To generate a numpy array from correlation with true or false
mask[np.triu_indices_from(mask)] = True # To have the index of the upper triangle
# Setup the matplotlib figures
f, ax = plt.subplots(figsize = (20,10))
f.suptitle('Correlation Matrix', fontsize=40)
# Generate a custum diverging color map
cmap = sns.diverging_palette(10, 0, as_cmap=True)
# Draw the heatmap with the mask and the correct aspect ratio
sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmax=1, center=0,
            square=True, linewidth=5, cbar_kws={'shrink': .5})

"""## Feature engineering - Response"""

dataset.dtypes
# The dates are objects, we need to change it. One reason is, we can calculate the difference between the first open and the enrolled date.
# This differece can be visualised in a distribution.

"""### Transforming the dates into numerical dates"""

dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]
dataset.dtypes

# Selecting Time For Response
dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')

response_hist = plt.hist(dataset['difference'].dropna(), color= '#3F5D7D')
plt.title('Distribution of Time-Since-Screan-Reached')

plt.hist(dataset['difference'].dropna(), color= '#3F5D7D', range = [0,100])
plt.title('Distribution of Time-Since-Screem-Reached')

"""As we can see in the above figure, we do not have a great number of inscription after 100 hours of application usage. In this case, to make the model more clear, we are going to change the response, all custumer that toke more then 48 hour to subscription will be turned into 0, it means, we do not consider them as a possible custumer.

### Changing the response based on Distribution of Time-Since-Screem-Reached
"""

dataset.loc[dataset.difference > 48, 'enrolled'] = 0

# Now, one importante thing is to eliminate certains columns in our dataset. Due to the change on the response variable, the features first_open, date_enrolled and difference there are no more sense, for this reason we drop theese columns
dataset = dataset.drop(columns = ['difference', 'first_open', 'enrolled_date'])

"""## Feature engineering on the screem variables

### Creating top_screens columns
"""

# Loading the second csv file
top_screens = pd.read_csv('top_screens.csv').top_screens.values

# We need to change the separator of screen_list column
dataset['screen_list'] = dataset.screen_list.astype(str) +','

# Putting the top_screens as columns

for sc in top_screens:
  dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
  dataset['screen_list'] = dataset.screen_list.str.replace(sc+',','')

"""### Creating Other Column"""

# We have more than 15 screens, so we need to count the others screens are not in the top_screens list

dataset['Other'] = dataset.screen_list.str.count(',')

# Now, we do not more need the screen_list

dataset = dataset.drop(columns=['screen_list'])

"""## Funnels (Correleted Screens)"""

saving_screens = ['Saving1',
                  'Saving2',
                  'Saving2Amount',
                  'Saving4',
                  'Saving5',
                  'Saving6',
                  'Saving7',
                  'Saving8',
                  'Saving9',
                  'Saving10',]

dataset['TotalSaving'] = dataset[saving_screens].sum(axis = 1) # creating a new column

# Eliminating the saving_screens columns
dataset = dataset.drop(columns=saving_screens)

cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

dataset.head()

dataset.describe()

dataset.columns

"""## Saving the dataset"""

dataset.to_csv('new_appdata10.csv', index = False)

"""# Model Processing

## Data prepocessing
"""

# Importing the new dataset
dataset_model = pd.read_csv('new_appdata10.csv')

# We need to separete the response column of the rest dataframe
response = dataset_model['enrolled']
# Eliminating the enrolled column from dataset_model
dataset_model = dataset_model.drop(columns='enrolled')

# Splitting the Dataset into Testing and Training variables

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_model, response, test_size = 0.2, random_state = 0)

# We had preserved the user column, but this does not have signification to model. We are going to save this column into other variable. In the future we can associate the user with the prediction
train_id = X_train['user']
X_train = X_train.drop(columns='user')
test_id = X_test['user']
X_test = X_test.drop(columns='user')

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_bckp = pd.DataFrame(sc.fit_transform(X_train))
X_test_bckp = pd.DataFrame(sc.transform(X_test))
X_train_bckp.columns = X_train.columns.values
X_test_bckp.columns = X_test.columns.values
X_train_bckp.idex = X_train.index.values
X_test_bckp.idex = X_test.index.values
X_train = X_train_bckp
X_test = X_test_bckp

# Model building

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

## Predicting the test result

y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)
y_pred4 = classifier4.predict(X_test)
y_pred5 = classifier5.predict(X_test)
y_pred6 = classifier5.predict(X_test)
y_pred7 = classifier7.predict(X_test)

#Selection Model

from sklearn.metrics import classification_report, confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
cm4 = confusion_matrix(y_test, y_pred4)
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