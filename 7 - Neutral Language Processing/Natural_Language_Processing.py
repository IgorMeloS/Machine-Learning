# ===================================================================================
# Natural Language Processing in Python
# This program has as objective to construct a Bag of Words Model a branch of
# Natural Language Processing.
T# The data-set is composed by many reviews of quality and service of a restaurant.
# The structure of the data-set is two columns. The first column contains the review
# and the second the sentiment analysis (0 to negative sentiment and 1 to positive
# sentiment). We are going to train the model to predict the sentiment of a review.
# ===================================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
# This data-set is composed by two columns and one thousand rows (number of reviews).


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# Here we employed the delimiter to read a tabular separate and quoting = 3 to suppress all quotes (it can interfere in the model)

# Cleaning the Text

# An important step in Natural Language Processing is to clean your text. It's means that we are going to select the relevant words to take account in our model. For example, words like I, you, Me, this, that, a, an are not relevant to the model. For this reason we utilize Natural Language Toolkit that provides a list with the most non  relevant words and from this list we can eliminate these words from our list of relevant words.
# An other important process is to stemmer the words, it’s means we just consider the root of the words, for example, loved, lovely will be stemmed to love.
# One apply it to reduce our sparse matrix. So for the text cleaning we utilize some modulus as NLTK and RE.


import re # Regular Expression
import nltk # Natural Language Toolkit
nltk.download('stopwords') # Stopwords is a list that contains all not relevant words
from nltk.corpus import stopwords # Calling Stopwords class to creat the list with non relevant words
from nltk.stem.porter import PorterStemmer # The class to stemmer the words
corpus = [] # Empty list. We are going to add the cleaned text into this list
# This loop for to clean the text row by row
for i in range(0, 1000):
  # Now we creat a new variable called review, we transform the original text, first we elimate all pontuations and then all capital letters
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # This eliminates all potuation the first [] refers to the column and the second [] to the rows
  review = review.lower() # To transform capital letter into lowercase
  review = review.split() # We need to split the elements to create a list with many elements, in the past step we had just a list with one single element
  ps = PorterStemmer() # The objecto to stemmer the words
  all_stopwords = stopwords.words('english') # List that contains all stopwords (non relevant words)
  all_stopwords.remove('not') # The word not is inside the stop word, but for this sentiment analysis it is important. We remove from the stopwords list
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # Here eliminate the stopwords from review and then we transform to the root form
  review = ' '.join(review) # We join all the words in each row to respect the original format
  corpus.append(review) # Here we create the final list that contians the cleaned text and from this list we are going to construct our model.

# Creating the Bag of Words Model

# This model transforms words into to number and construct a sparse matrix. This matrix contains our independent variables (numbers that represent each word
# according with you frequency in the text, for example a word that appears two times will have 2 as associate number. Each word will be considered as a
# column. If a certain word appears in a certain row (text review), in this row we are going to see a number different of zero). Once time that we have the
# sparse matrix we are able to train our model of sentiment analysis.

from sklearn.feature_extraction.text import CountVectorizer # This class transform words into a number
cv = CountVectorizer(max_features=1500) # This object converts words into number (max_feature is the number of desired word (number of columns of the sparse matrix))
X = cv.fit_transform(corpus).toarray() # Here we create our sparse matrix and we transform words into numbers (Tokenization)
y = dataset.iloc[:, -1].values # Here we create our dependent variable from the dataset

# Splitting Dataset into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training all classification models

# The response to know a new review come from a classification model.
# Here we have seven different models of classification to train the data-set.

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

# Making the Test Results

#Here we predict the test results for each trained model

y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)
y_pred4 = classifier4.predict(X_test)
y_pred5 = classifier5.predict(X_test)
y_pred6 = classifier5.predict(X_test)
y_pred7 = classifier7.predict(X_test)

# Making Confusion Matrix and Accuracy Score

# Is important to verify the accuracy score to choose the more apropriate model for this dataset.

from sklearn.metrics import confusion_matrix, accuracy_score
cm1 = confusion_matrix(y_test, y_pred1)
ac1 = accuracy_score(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
ac2 = accuracy_score(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
ac3 = accuracy_score(y_test, y_pred3)
cm4 = confusion_matrix(y_test, y_pred4)
ac4 = accuracy_score(y_test, y_pred4)
cm5 = confusion_matrix(y_test, y_pred5)
ac5 = accuracy_score(y_test, y_pred5)
cm6 = confusion_matrix(y_test, y_pred6)
ac6 = accuracy_score(y_test, y_pred6)
cm7 = confusion_matrix(y_test, y_pred7)
ac7 = accuracy_score(y_test, y_pred7)
print('Checking Confusion Matrix and accuracy score to a single observation')
print('Logistic Regression Classification')
print(cm1)
print(ac1)
print('\n')
print('K Nearest Neighbors')
print(cm2)
print(ac2)
print('\n')
print('Support Vector Machine')
print(cm3)
print(ac3)
print('\n')
print('Kernel Support Vector Machine')
print(cm4)
print(ac4)
print('\n')
print('Naive Bayes Classification')
print(cm5)
print(ac5)
print('\n')
print('Decision Tree Classification')
print(cm6)
print(ac6)
print('\n')
print('Random Forest Classification')
print(cm7)
print(ac7)
print('\n')

# Computing the accuracy with k-Fold Cross Validation

# K-Fold Cross Validation gives a mean of accuracy for each model, it's one way to better choose the most appropriate model.
# Here we consider the accuracy mean from 10 repetition.

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
print("Accuracy: {:.2f} %".format(accuracies7.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies7.std()*100))

# Predict a Single review Positive and Negative

# Here we make a single prediction to a positive and a negative review.
# We chose Random Forest Classification because this model presents an accuracy
# score of 78.50% and a Standard Deviation of 2.08%. We have models with high
# accuracy in comparison with Random Forest Classification, as Logistic Regression,
# for example. But the Standard Deviation of Random Forest Classification is more satisfactory.

# Positive case
new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier7.predict(new_X_test)
if new_y_pred == 1:
  print('Positive Sentiment')
else:
  print('Negative Sentiment')

# Negative case
new_review = 'I hate this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier7.predict(new_X_test)
if new_y_pred == 1:
  print('Positive Sentiment')
else:
  print('Negative Sentiment')

# ========================================================================================
# Conclusion
# We worked with a Bag of Words Model and we obtained a satisfactory result.
# To have a good results we must to make attention in the cleaning text process, because
# it is one of the most important step in this model. After, we must to choose the better
# model to train.
# With these models we were able to create good responses with respect to sentiment
# analysis (yes or no / 0 or 1).
# ========================================================================================
