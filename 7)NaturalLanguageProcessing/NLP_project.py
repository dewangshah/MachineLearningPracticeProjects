import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## The Data

yelp=pd.read_csv('yelp.csv')

print(yelp.head())
print(yelp.info())
print(yelp.describe())

yelp['text length']=yelp['text'].apply(len)

# EDA

#Using FacetGrid from the seaborn library, create a grid of 5 histograms of text length based off of the star ratings.
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
plt.show()

#Create a boxplot of text length for each star category.
sns.boxplot(x='stars',y='text length',data=yelp)
plt.show()

#Create a countplot of the number of occurrences for each type of star rating.
sns.countplot(x='stars',data=yelp)
plt.show()

yelp.groupby('stars').mean()
yelp.groupby('stars').mean().corr()

#heatmap based off that .corr() dataframe
sns.heatmap(yelp.groupby('stars').mean().corr(),cmap='coolwarm',annot=True)
plt.show()

## NLP Classification Task

#Creating a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.
yelp_class=yelp[(yelp['stars'] ==1) | (yelp['stars']==5)]

from sklearn.model_selection import train_test_split
X=yelp_class['text']
y=yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(X)

## Train Test Split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

## Training a Model

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

nb.fit(X_train,y_train)

## Predictions and Evaluations

predictions=nb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Using Text Processing

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

pipeline=Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

## Using the Pipeline

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)

### Predictions and Evaluation

predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Looks like Tf-Idf actually made things worse!
