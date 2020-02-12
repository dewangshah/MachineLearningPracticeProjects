'''
This project explores the publicly available data from LendingClub.com.
Lending Club connects people who need money (borrowers) with people who have money (investors).
Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability
of paying you back. This model helps predict this.

Lending data from 2007-2010 was used to classify and predict whether or not the borrower paid back their loan in full. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Get the Data

loans=pd.read_csv('loan_data.csv')

print(loans.info())
print(loans.describe())
print(loans.head())

# Exploratory Data Analysis

#Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=30)
loans[loans['credit.policy']==0]['fico'].hist(bins=30)
plt.show()

#Similar figure, except this time select by the not.fully.paid column
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30)
loans[loans['not.fully.paid']==1]['fico'].hist(bins=30)
plt.show()

sns.countplot(x='purpose',data=loans,hue='not.fully.paid')
plt.show()
sns.jointplot(x='fico',y='int.rate',data=loans)
plt.show()

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid')
plt.show()

# Setting up the Data

loans.info()

## Categorical Features

cat_feats=loans['purpose']

purpose_data = pd.get_dummies(loans['purpose'],drop_first=True)
final_data=pd.concat([loans,purpose_data],axis=1)
final_data.drop('purpose',axis=1,inplace=True)

final_data.head()

## Train Test Split

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

## Training a Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

## Predictions and Evaluation of Decision Tree

predictions=dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

## Training the Random Forest model

from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier(n_estimators=600)
rdf.fit(X_train,y_train)

## Predictions and Evaluation

rdf_pred=rdf.predict(X_test)

print(confusion_matrix(y_test,rdf_pred))
print('\n')
print(classification_report(y_test,rdf_pred))

#Random forest performs better with an accuracy of 0.81 as compared to decision tree with an accuracy of 0.76
