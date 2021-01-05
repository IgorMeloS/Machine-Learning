#!/usr/bin/env python
# coding: utf-8

# **Directing Customers to Subscription Through App Behavior Analysis**
# 
# This notebook wants to build a model to predict if a certain client will subscribe or not a premium version of a mobile application.
# 
# 
# **Problem Description**
# 
# Nowadays, many e-companies have application services, some companies offer completely free application and others have a premium version, it means, a paid version. For this case, we have a company who wants to know which clients will enroll to the premium version (in the trial version the services are limited). To build this model, we consider the customer behavior based on the utilization of a free version. It’s important to know that to each customer, the company offers 24 hours to try the premium version. The goal to build this model is to improve the marketing analysis. Once the company knows which clients will probably to enroll, it can direct your efforts to capture these customers. In this manner, the company can reduce the marketing spend and optimize the number of enrolled users.
# 
# **Dataset Description**
# 
# The dataset is built based on the app usage. The company stores the historical activities of each customer. The features for this dataset are
# 
# + User (identifier number)
# + First Open (day and hour, M/D and AM/PM)
# + Day of Week (0 to 6, Sunday to Saturday)
# + Hour (0 to 23)
# + Age
# + Screen List (the list of application services accessed by a client in 24 hours)
# + Number of screens (total number of services accessed by a client in 24 hours)
# + Mini game (0 or 1, this is a game offered by the application)
# + Liked (0 or 1, if a customer liked any service of the application in 24 hours)
# + Used Premium Features (0 or 1, if a client used the trial premium version)
# + Enrolled (target variable, 0 or 1)
# + Enrolled date (date and hour, M/D and AM/PM).
# 
# So, we have 12 features (columns) and 50000 instances (that represent each customer). Beyond this preliminaries information, we also have a dataset that contains the most accessed screens in 24 hours, this will be important in the feature engineering process.

# # Importing Libraries

# In[2]:


import pandas as pd
from dateutil import parser # to take care about time and date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


# # Importing Dataset

# In[3]:


dataset = pd.read_csv('appdata10.csv')


# # Data Preprocessing

# ## Visualizing some informations from the dataset

# In[4]:


dataset.shape


# In[5]:


dataset.dtypes


# In[6]:


dataset.head()


# In[7]:


dataset.describe()


# ### Taking cara about the hour variable
# 
# As we can see the values in the Hour column is an object type, we must to change into a numerical integer variable.

# In[8]:


dataset['hour'] = dataset.hour.str.slice(1,3).astype(int)


# In[9]:


dataset.describe()


# # Exploratory Data Analysis (EDA)

# ## Analyzing categorical variables

# ### Target Variable

# In[10]:


dataset['enrolled'].value_counts()


# In[11]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=dataset['enrolled'])
plt.subplot(1, 2, 2)
values = dataset.iloc[:, - 3].value_counts(normalize = True).values # to show the binirie values in parcentage
index = dataset.iloc[:, -3].value_counts(normalize = True).index
plt.pie(values, labels= index, autopct='%1.1f%%', colors=['b', 'tab:orange'])
plt.show()


# ### Categorical Features
# 
# The categorical features are mini game, used premium feature and liked. We are going to see the pier plot distribution and the count plot according with the target variable.

# In[12]:


categorical_list = ['minigame','used_premium_feature', 'liked', 'enrolled']


# In[13]:


data_cat = dataset[categorical_list]


# In[14]:


fig = plt.figure(figsize=(15, 15))
plt.suptitle('Pie Chart Distribution', fontsize = 20)
for i in range(1, data_cat.shape[1]):
    plt.subplot(2, 2, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(data_cat.columns.values[i - 1])
  # Setting the biniries values
    values = data_cat.iloc[:, i - 1].value_counts(normalize = True).values # to show the binirie values in parcentage
    index = data_cat.iloc[:, i -1].value_counts(normalize = True).index
    plt.pie(values, labels= index, autopct='%1.1f%%')
    plt.axis('equal')
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[15]:


plt.figure(figsize=(15, 10))
for i in range(1, data_cat.shape[1]):
    plt.subplot(2, 2, i)
    sns.countplot(x=data_cat.iloc[: , i-1], hue=data_cat['enrolled'])
plt.show()


# ## Analyzing numerical variables

# In[16]:


numerical_list = ['dayofweek', 'age', 'numscreens', 'enrolled']


# In[17]:


data_num  = dataset[numerical_list]


# In[18]:


plt.figure(figsize=(25,15))
plt.suptitle('Histograms of numerical variables (mean values)', fontsize = 20)
for i in range(1, data_num.shape[1]):
    plt.subplot(2, 2, i)
    f = plt.gca()
    sns.histplot(data_num.iloc[:, i-1], color = '#3F5D7D', kde= True)
plt.show()


# In[19]:


plt.figure(figsize=(25,15))
plt.suptitle('Histograms of numerical variables (mean values)', fontsize = 20)
for i in range(1, data_num.shape[1]):
    plt.subplot(2, 2, i)
    f = plt.gca()
    sns.histplot(data=data_num, x=data_num.iloc[:, i-1], hue='enrolled', kde = True)
plt.show()


# ## Correlation between the target and independent variables

# In[20]:


dataset.drop(columns='enrolled').corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
                                            title = 'Correlation with reponse variable',
                                            fontsize = 15, rot = 45, grid = True, color = '#5F5D7D')


# ## Correlation Matrix between all variables

# In[21]:


sns.set(style='white', font_scale= 1)
corr = dataset.corr() # here we compute the correlation between numericals variables
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


# # Feature engineering
# Explain Here

# In[22]:


dataset.dtypes
# The dates are objects, we need to change it. One reason is, we can calculate the difference between the first open and the enrolled date.
# This differece can be visualised in a distribution.


# ## Feature engineering to age variable

# def category_age(age):
#     if (age <=20):
#         return 0
#     elif(age > 20) & (age <= 30):
#         return 10
#     elif(age > 30) & (age <= 40):
#         return 5
#     else:
#         return 1

# dataset['age'] = dataset['age'].map(category_age)

# ## Feature engineering to numscreen variable

# def category_ns(numscreens):
#     if (numscreens  <=15):
#         return 0
#     elif(numscreens > 15) & (numscreens <= 35):
#         return 10
#     elif(numscreens > 35) & (numscreens <= 55):
#         return 5
#     else:
#         return 1

# dataset['numscreens'] = dataset['numscreens'].map(category_age)

# ## Feature engineering to transform the date into delta time variables and change the response variable
# 
# The features first open and enrolled date are object variables, to build our model this kind of variable are not convenient.

# In[23]:


dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]] # parser transforms date object into delta time variable


# In[24]:


dataset["first_open"]


# In[25]:


dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]
# the if condition takes care about the NaN values


# In[26]:


dataset["enrolled_date"]


# ### Creating a delta time variable
# 
# This variable returns the time difference between the first access and the enrolled date.

# In[27]:


# Selecting Time For Response
dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')


# In[28]:


dataset["difference"]


# ### Visualizing the distribution of delta time variable

# In[29]:


plt.figure()
plt.title('Distribution of Time-Since-Screem-Reached')
plt.hist(dataset['difference'].dropna(), color= '#3F5D7D', range = [0,100])
plt.show()


# As we can see in the figure , the distribution does not show a great number of inscription between 20 and 100 hours of application usage. In this case, to make the model more clear, we are going to change the response, all customer which made the  subscription  after 24 hour will be turned into 0, it means, we do not consider them as a possible customer.

# ### Changing the response based on Distribution of Time-Since-Screem-Reached

# In[30]:


dataset.loc[dataset.difference > 24, 'enrolled'] = 0


# Now, one important thing is to eliminate certain columns in our dataset. Due to the change on the response variable, the features first_open, date_enrolled and difference do not have more sense, for this reason, we drop these columns.

# In[31]:


dataset = dataset.drop(columns = ['difference', 'first_open', 'enrolled_date'])


# ## Feature engineering on the screem variables
# 
# The analysis of the screen (this is the service that the customer had visited) will be a very important step to build the model. Since we know which service each client had accessed, this information must be transformed into numerical values. The objective is to create a column to each the most visited screens and count how many times a client visited it.
# 
# Fortunately, we have the information about the most visited screen in a period of 24 hours from a second dataset. We are going to consider these screen to create a set of columns, the others screen which are not present in this list will be replaced into an other column called ‘other’.

# ### Creating the most visited screens columns

# In[32]:


# Loading the second csv file
top_screens = pd.read_csv('top_screens.csv').top_screens.values


# In[33]:


top_screens.shape # we have 58 top screens


# In[34]:


top_screens


# In[35]:


dataset['screen_list'] = dataset.screen_list.astype(str) +',' 
# We need to change the separator of screen_list column due to count of each screen that will be realized


# In[36]:


# Putting the top_screens as columns and count the frequency to each customer
for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int) # creating the columns to each top screen and counting the frequency to each client
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+',','') # replacing the top screen by '' in the screen list column 


# ### Creating the column 'other'
# 
# Here we count all screens whice are not included in the top screen file.

# In[37]:


dataset['Other'] = dataset.screen_list.str.count(',')


# In[38]:


# Now, we do not more need the screen_list
dataset = dataset.drop(columns=['screen_list'])


# ### Funnels (Correleted Screens)
# Explain what is this funnels and try to find a good explanation.

# In[39]:


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


# In[40]:


dataset['TotalSaving'] = dataset[saving_screens].sum(axis = 1) # creating a new column


# In[41]:


# Eliminating the saving_screens columns
dataset = dataset.drop(columns=saving_screens)


# In[42]:


cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)


# In[43]:


cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)


# In[44]:


loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)


# # EDA for the expanded dataset

# ## Visualizing some dataset information

# In[45]:


dataset.shape


# In[46]:


dataset.head()


# In[47]:


dataset.iloc[: , 0: 15].describe()


# In[48]:


dataset.iloc[: , 15: 30].describe()


# Here, we can see that the feature screen Profil Children does not have access, showing a mean value equal to zero. We do not need to consider this variable to build the model.

# In[49]:


dataset.iloc[: , 30: 45].describe()


# In[50]:


dataset.iloc[: , 45: ].describe()


# In[53]:


dataset.columns


# In[54]:


del dataset['ProfileChildren ']


# In[55]:


dataset.shape


# ## Correlation among screen variables and the target

# In[56]:


dataset.iloc[:, 8:26].corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
                                            title = 'Correlation with reponse variable',
                                            fontsize = 15, rot = 45, grid = True, color = '#5F5D7D')


# In[57]:


dataset.iloc[:, 26:].corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
                                            title = 'Correlation with reponse variable',
                                            fontsize = 15, rot = 45, grid = True, color = '#5F5D7D')


# ## Matrix correlation between all variables

# In[58]:


plt.figure()
sns.set(style='white', font_scale= 1)
corr = dataset.corr() # here we compute the correlation between numericals variables
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype= np.bool) # To generate a numpy array from correlation with true or false
mask[np.triu_indices_from(mask)] = True # To have the index of the upper triangle
# Setup the matplotlib figures
f, ax = plt.subplots(figsize = (20,10))
f.suptitle('Correlation Matrix', fontsize=40)
# Generate a custum diverging color map
cmap = sns.diverging_palette(10, 0, as_cmap=True)
# Draw the heatmap with the mask and the correct aspect ratio
sns.heatmap(corr, mask=mask, annot=False, cmap=cmap, vmax=1, center=0,
            square=True, linewidth=5, cbar_kws={'shrink': .5})
plt.show()


# ## Matrix correlation between screen variables

# In[59]:


plt.figure()
sns.set(style='white', font_scale= 1)
corr = dataset.iloc[:, 8:].corr() # here we compute the correlation between numericals variables
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype= np.bool) # To generate a numpy array from correlation with true or false
mask[np.triu_indices_from(mask)] = True # To have the index of the upper triangle
# Setup the matplotlib figures
f, ax = plt.subplots(figsize = (50,40))
f.suptitle('Correlation Matrix', fontsize=40)
# Generate a custum diverging color map
cmap = sns.diverging_palette(10, 0, as_cmap=True)
# Draw the heatmap with the mask and the correct aspect ratio
sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmax=1, center=0,
            square=True, linewidth=5, cbar_kws={'shrink': .5})
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Saving the dataset

# In[47]:


dataset.to_csv('new_appdata10.csv', index = False)


# # Model Processing

# ## Data prepocessing

# ### Importing the new dataset

# In[48]:


dataset_model = pd.read_csv('new_appdata10.csv')


# In[49]:


dataset_model.shape


# In[50]:


dataset_model.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


# We need to separete the response column of the rest dataframe
response = dataset['enrolled']
# Eliminating the enrolled column from dataset_model
i_var = dataset.drop(columns='enrolled')


# ### Splitting the Dataset into Testing and Training variables

# In[61]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(i_var, response, test_size = 0.2, random_state = 0)


# In[62]:


# We had preserved the user column, but this does not have signification to model. We are going to save this column into other variable. In the future we can associate the user with the prediction
train_id = X_train['user']
X_train = X_train.drop(columns='user')
test_id = X_test['user']
X_test = X_test.drop(columns='user')


# ### Feature scaling

# In[63]:


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


# ## Models building

# In[64]:


# Logistic Regression

t0 = time.time()
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0, penalty='l2')
classifier1.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))
# K Nearest Nieghbors

t0 = time.time()
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p = 2)
classifier2.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Support Vector Machine - Linear Classifier

#t0 = time.time()
#from sklearn.svm import SVC
#classifier3 = SVC(kernel= 'linear', random_state=0)
#classifier3.fit(X_train, y_train)
#t1 = time.time()
#print("Took %0.2f seconds" % (t1 - t0))

# Kernel Support Vector Machine

#t0 = time.time()
#from sklearn.svm import SVC
#classifier4 = SVC(kernel='rbf', random_state=0)
#classifier4.fit(X_train, y_train)
#t1 = time.time()
#print("Took %0.2f seconds" % (t1 - t0))

# Naïves Bayes Classification
t0 = time.time()
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Decision Tree Classification
t0 = time.time()
from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier6.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Random Forest Classification
t0 = time.time()
from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
classifier7.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


# In[65]:


## Predicting the test result
t0 = time.time()
y_pred1 = classifier1.predict(X_test)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

t0 = time.time()
y_pred2 = classifier2.predict(X_test)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

#y_pred3 = classifier3.predict(X_test)
#y_pred4 = classifier4.predict(X_test)
t0 = time.time()
y_pred5 = classifier5.predict(X_test)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

t0 = time.time()
y_pred6 = classifier6.predict(X_test)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

t0 = time.time()
y_pred7 = classifier7.predict(X_test)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


# ## Selection Model

# In[66]:


from sklearn.metrics import classification_report, confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
#cm3 = confusion_matrix(y_test, y_pred3)
#cm4 = confusion_matrix(y_test, y_pred4)
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
print('Support Vector Machine - Not selected')
#print(cm3)
#print(classification_report(y_test,y_pred3))
print('\n')
print('Kernel Support Vector Machine - Not selected')
#print(cm4)
#print(classification_report(y_test,y_pred4))
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


# In[67]:


from sklearn.model_selection import cross_val_score
accuracies1 = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
#accuracies2 = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 10)
#accuracies3 = cross_val_score(estimator = classifier3, X = X_train, y = y_train, cv = 10)
#accuracies4 = cross_val_score(estimator = classifier4, X = X_train, y = y_train, cv = 10)
accuracies5 = cross_val_score(estimator = classifier5, X = X_train, y = y_train, cv = 10)
accuracies6 = cross_val_score(estimator = classifier6, X = X_train, y = y_train, cv = 10)
accuracies7 = cross_val_score(estimator = classifier7, X = X_train, y = y_train, cv = 10)

print('Checking K-Fold Cross Validation')
print('\n')
print('Logistic Regression Classification')
print("Accuracy: {:.2f} %".format(accuracies1.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies1.std()*100))
print('\n')
#print('K Nearest Neighbors')
#print("Accuracy: {:.2f} %".format(accuracies2.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies2.std()*100))
#print('\n')
#print('Support Vector Machine')
#print("Accuracy: {:.2f} %".format(accuracies3.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies3.std()*100))
#print('\n')
#print('Kernel Support Vector Machine')
#print("Accuracy: {:.2f} %".format(accuracies4.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies4.std()*100))
#print('\n')
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


# In[68]:


X_train.shape


# ## Reducing

# In[ ]:


from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=24, kernel='rbf')
X_p_train = kpca.fit_transform(X_train)
X_p_test = kpca.transform(X_test)


# In[ ]:


# Logistic Regression

t0 = time.time()
from sklearn.linear_model import LogisticRegression
classifier1p = LogisticRegression(random_state = 0, penalty='l2')
classifier1p.fit(X_p_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))
# K Nearest Nieghbors

t0 = time.time()
from sklearn.neighbors import KNeighborsClassifier
classifier2p = KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p = 2)
classifier2p.fit(X_p_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Support Vector Machine - Linear Classifier

#t0 = time.time()
#from sklearn.svm import SVC
#classifier3 = SVC(kernel= 'linear', random_state=0)
#classifier3.fit(X_p_train, y_train)
#t1 = time.time()
#print("Took %0.2f seconds" % (t1 - t0))

# Kernel Support Vector Machine

#t0 = time.time()
#from sklearn.svm import SVC
#classifier4 = SVC(kernel='rbf', random_state=0)
#classifier4.fit(X_p_train, y_train)
#t1 = time.time()
#print("Took %0.2f seconds" % (t1 - t0))

# Naïves Bayes Classification
t0 = time.time()
from sklearn.naive_bayes import GaussianNB
classifier5p = GaussianNB()
classifier5p.fit(X_p_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Decision Tree Classification
t0 = time.time()
from sklearn.tree import DecisionTreeClassifier
classifier6p = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier6p.fit(X_p_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

# Random Forest Classification
t0 = time.time()
from sklearn.ensemble import RandomForestClassifier
classifier7p = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
classifier7p.fit(X_p_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Boosting the model
# 

# In[ ]:


param_grid1 = {'C' : [0.001, 0.01, 0.1, 1 , 10, 100], 'solver' : ['newton-cg', 'sag', 'saga','lbfgs', 'liblinear' ], 'penalty' : ['l1', 'l2']}
param_grid2 = {'criterion': ['gini', 'entropy'], 'n_estimators': [10, 50, 100, 500, 1000]}


# In[ ]:


from sklearn.model_selection import GridSearchCV

grid1 = GridSearchCV(classifier1, param_grid1, cv = 10, verbose = 4)
grid2 = GridSearchCV(classifier7, param_grid2, cv = 10, verbose = 4)


# class_boost_1 = grid1.fit(X_train,y_train)
# class_boost_2 = grid2.fit(X_train,y_train)

# boost_predictions1 = class_boost_1.predict(X_test)
# boost_predictions2 = class_boost_2.predict(X_test)

# cmb1 = confusion_matrix(y_test, boost_predictions1)
# cmb2 = confusion_matrix(y_test, boost_predictions2)

# sns.heatmap(cmb1, annot=True)
# print(classification_report(y_test,boost_predictions1))
# print('The best parameters for Logistic Regression:')
# print(grid1.best_params_)
# print('\n')
# print('Confusion Matrix')

# print('Average Accuracy {:.2f}%'.format(grid1.best_score_ * 100))
# print('Standard Deviation {:.2f}%'.format(grid1.cv_results_['std_test_score'][grid1.best_index_] * 100))

# sns.heatmap(cmb2, annot=True)
# print(classification_report(y_test,boost_predictions2))
# print('Random Forest Classification:')
# print(grid2.best_params_)
# print('\n')
# print('Confusion Matrix')

# print('Average Accuracy {:.2f}%'.format(grid2.best_score_ * 100))
# print('Standard Deviation {:.2f}%'.format(grid2.cv_results_['std_test_score'][grid2.best_index_] * 100))

# # Conclusions
# Here I need to explain in a good way what are the conclusions
