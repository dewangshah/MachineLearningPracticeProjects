

import pandas as pd
df=pd.read_csv('bank_note_data.csv')

print('Original Data:\n',df.head())


## EDA

import seaborn as sns
import matplotlib.pyplot as plt

#Create a Countplot of the Classes (Authentic 1 vs Fake 0)
sns.countplot(x='Class', data=df)
plt.show()

#Create a PairPlot of the Data with Seaborn, set Hue to Class **
sns.pairplot(data=df, hue='Class')
plt.show()

## Data Preparation 

"""
When using Neural Network and Deep Learning based systems, it is usually a good idea to Standardize your data,
this step isn't actually necessary for our particular data set
"""

### Standard Scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(df.drop('Class',axis=1))
scaled_features=scaler.transform(df.drop('Class',axis=1))

df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

## Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df_feat,df['Class'],test_size=0.3)

# Tensorflow

import tensorflow as tf

feat_cols=[]
for col in X_train.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
    
feat_cols

classifier=tf.estimator.DNNClassifier(hidden_units=[10,20,10],n_classes=2,feature_columns=feat_cols)

input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,num_epochs=5,shuffle=True)
classifier.train(input_fn=input_func, steps=500)

## Model Evaluation

pred_fn=tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
predictions=list(classifier.predict(input_fn=pred_fn))

final_preds=[]
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

from sklearn.metrics import confusion_matrix,classification_report
print('DNNClassifier Confusion Matrix:\n',confusion_matrix(y_test,final_preds))
print('DNNClassifier Classification Report:\n',classification_report(y_test,final_preds))


## Comparing DNNClassifier Vs RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred=rfc.predict(X_test)

print('RandomForestClassifier Confusion Matrix:\n',confusion_matrix(y_test,rfc_pred))
print('RandomForestClassifier Classification Report:\n',classification_report(y_test,rfc_pred))
