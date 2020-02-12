
"""With a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement,
the code creates a model that will predict whether or not they will click on an ad based off the features of that user"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Get the Data

ad_data=pd.read_csv('advertising.csv')

print(ad_data.head())
print(ad_data.describe())
print(ad_data.info())

## Exploratory Data Analysis

plt.hist(ad_data['Age']) #A histogram of the Age
plt.show()
sns.jointplot(x='Age',y='Area Income', data=ad_data) #A jointplot showing Area Income versus Age
plt.show()
sns.jointplot(x='Age',y='Daily Time Spent on Site', data=ad_data, kind='kde') #A jointplot showing the kde dist. of Daily Time spentvs. Age
plt.show()
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage', data=ad_data) #A jointplot of Daily Time Spent on Site vs.Daily Internet Usage
plt.show()
sns.pairplot(ad_data,hue='Clicked on Ad') #A pairplot with the hue defined by the Clicked on Ad column feature
plt.show()

# Logistic Regression

from sklearn.model_selection import train_test_split
X=ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y=ad_data['Clicked on Ad']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101) #Split the data into training set and testing set using train_test_split

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)

## Predictions and Evaluations

predictions=logmodel.predict(X_test) #Predicting values for the testing data

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions)) #Creating a classification report for the model
