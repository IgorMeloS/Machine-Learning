# =============================================================================
# Assocition Rules in Python Using Apriori Algororithm
# Program wrote by Igor Melo.
# This program has as objective to construct association rules between the 
# products in a grocery store. Basically, we want to know if someone buy a 
# bottle of milk what are the others products which can be purchased.
# =============================================================================

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
!pip install apyori

## Importing Dataset

# Data Preprocessing

#In this problem our data set is composed by many costumers transaction. So in 
# structure of data set rows represent costumers and columns products bought. 
# For this reason we must to declare the parameter header=None, the columns not 
# have a label association. Other aspect of this kind of problem is the necessity 
# to make a list of translations because it is more appropriate to the Apryori 
# library. We must also transform our values in string type. To construct our 
# list we use for loop. Apart from that list we are going to train our model.

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the Apriori model on the dataset

# Apriori method is a very good way to make association rules. To construct the 
# model we need to know three important factors. Support, Confidence and Lift.

# Support is the product that we analyses, for example, we have 100 transaction 
# in a grocery store of which 20 are cheese purchase. So the Support for the cheese 
# purchase is S(cheese) = 20/100 = 0.2%.

# Confidence is the value that call us the likelihood of someone that bought cheese 
# bought also bread. Imagine that 8 person bought both products, we write the 
# confidence as C(bread|cheese) = (Bread and cheese)/ (Total number of bread purchase). 
# We assume that 12 persons bought bread in this grocery store, so the confidence 
# has this values C(bread|cheese) = 8/12 = 0.66%.

# Lift is the ratio between the confidence and support defined as 
# L(cheese|bread) = C(cheese|bread)/S(cheese) = 0.66/0.2 = 3.3. In an Apriori 
# method, Lift is the most import value. The higher is better.

# In this simple analyses someone that buy cheese can buy bread

from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.004, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#We utilize the apyori object and the apriori class. The arguments are min_support, 
#min_confidence, min_lift, min_length and max_length. To choose the values of each 
# parameters we must to know the specificity of the problem. In this data set we 
#have the transaction of one week, so we consider that a product was bought 5 times 
# in a day, so we multiply for 7 days and we have 5*7= 35, so we devise by the total 
# transactions, min_support = 35/7501 = 0.004. For the min_confidence we chose 0.2. 
#Lift is the more important parameters, so above 3 we have a good rule. min and 
# max_length is the number of product in a rule, here we chose 2 to a pedagogical propose.

## Visualising the results

# Results coming directly from the apriori function

results = list(rules)
results

# Putting the results well organised

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Here we created a way to better visualize the results. But it's very specific 
#to this problem.
# In summary we take the elements by your index inside the results list.

# Displaying the results non sorted

resultsinDataFrame

# Displaying the results sorted by descending lifts

resultsinDataFrame.nlargest(n = 10, columns = 'Lift')

# =======================================================================================
# Here we have an example of association rule by the Apriori algorithm. The implantation 
# evolves some steps that can be made directly.
#
# In this example we treated a data set with many costumers transactions in a grocery 
# store and from this data we obtained the association rules between the products.
# We can see that for this grocery store people who bought Fromage Blanc can buy 
# also Honey given the lift value. We can see the same for the costumers who bought 
# Light Cream can buy also Chicken and so on.
#
# The results from this method aid to take decision, for example, how to organize the 
#products in supermarket aisle given the association rules.
# =======================================================================================



