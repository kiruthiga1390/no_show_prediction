#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:21:40 2018

@author: Kiruthiga
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


#load data
data = pd.read_csv('/Users/arunprasaath/Documents/kiruthiga/Quarter3/ml/KaggleV2-May-2016 2.csv')

data.head()


#rename columns
data = data.rename(columns={'PatientId':'patient_id','AppointmentID':'appointment_id','Gender': 'sex', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 
                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 
                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 
                                    'Handcap': 'handicap', 'No-show': 'no_show'})
data.head()


#binarize columns and plotting each field with no_show
data['no_show'] = data['no_show'].map({'No': 1, 'Yes': 0})
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
data['handicap'] = data['handicap'].apply(lambda x: 2 if x > 2 else x)

#get data and time
data['scheduled_day'] = pd.to_datetime(data['scheduled_day'], infer_datetime_format=True)
data['appointment_day'] = pd.to_datetime(data['appointment_day'], infer_datetime_format=True)

#remove outliers in age
data.drop(data[data['age'] <= 0].index, inplace=True)
data.drop(data[data['age'] >100].index, inplace=True)

data.describe()
data.head()

#encode neighbourhood
encoder_neighbourhood = LabelEncoder()
data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(data['neighbourhood'])

#Add new columns
data['waiting_time'] = list(map(lambda x: x.days+1 , data['appointment_day'] - data['scheduled_day']))
data.drop(data[data['waiting_time'] <= -1].index, inplace=True)

data['waiting_time_range'] = data['waiting_time'].apply(lambda x: 1 if x>=0 and x<=30 else 
                                                          2 if x>30 and x<=60 else 
                                                          3 if x>60 and x<=90 else 
                                                          4 if x>90 and x<=120 else 
                                                          5 if x>120 and x<=150 else
                                                          6 if x>150 and x<=180 else
                                                          7)
data['age_group'] = data['age'].apply(lambda x: 1 if x>0 and x<19 else 
                                                            2 if x>18 and x<38 else 
                                                            3 if x>37 and x<56 else 
                                                            4 if x>55 and x<76 else 5)

data['insurance_age'] = data['age'].apply(lambda x: 1 if x >= 65 else 0)

data['appointment_dayofWeek'] = data['appointment_day'].map(lambda x: x.dayofweek)

data['no_of_noshows'] = data.groupby('patient_id')[['no_show']].transform('sum')
data['total_appointment'] = data.groupby('patient_id')[['no_show']].transform('count')

data['risk_score'] = data.no_of_noshows / data.total_appointment


#visualization with no-show and show count
sns.countplot(x='sex', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='handicap', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='alcoholic', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='hypertension', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='diabetic', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='scholarship', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='age_group', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='insurance_age', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='appointment_dayofWeek', hue='no_show', data=data, palette='RdBu')
plt.show();
sns.countplot(x='waiting_time_range', hue='no_show', data=data, palette='RdBu')
plt.show();





#drop columns which are not contributing for prediction
data.drop(['scheduled_day','appointment_day','neighbourhood','patient_id','appointment_id','age_group','insurance_age','waiting_time_range'], axis=1, inplace=True)

#split data into train and test set
X = data.drop(['no_show'], axis=1)
y = data['no_show']
X.head()
y.head()
Counter(y)
sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_sample(X, y)
Counter(y_res)




X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=101)

#random classifier model

clf = RandomForestClassifier(n_estimators=300)

clf.fit(X_train, y_train)

#metrics of model

#accuracy
clf.score(X_test, y_test)
#confusion matrix
print(confusion_matrix(y_test, clf.predict(X_test)))
#classification report
print(classification_report(y_test, clf.predict(X_test)))

#features affecting prediction
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')