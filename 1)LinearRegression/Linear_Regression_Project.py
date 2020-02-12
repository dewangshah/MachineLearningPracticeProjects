
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Get the Data

customers=pd.read_csv('Ecommerce Customers')

print(customers.info())
print(customers.describe())

## Exploratory Data Analysis

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
plt.show()
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
plt.show()
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers,kind='hex')
plt.show()
sns.pairplot(customers)
plt.show()
sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)
plt.show()

## Training and Testing Data

X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
X.head()

y=customers['Yearly Amount Spent']
y.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression

## Training the Model

lm=LinearRegression()

lm.fit(X_train,y_train)

lm.coef_

## Predicting Test Data

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

## Evaluating the Model

from sklearn import metrics

print('MAE: ',metrics.mean_absolute_error(y_test,predictions))
print('MSE: ',metrics.mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(metrics.mean_absolute_error(y_test,predictions)))


## Residuals

plt.hist(y_test-predictions)

plt.show()

## Conclusion

cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(cdf)

#Based on the results, More focus should be given on the App
