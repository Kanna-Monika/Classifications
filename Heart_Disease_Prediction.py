# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 23:06:53 2022

@author: Monika
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset=pd.read_csv(r'C:\Users\Srinivas\Desktop\DS_Material\Feb_2022_Folder\16th,17th_Feb_2022\TASK-24\framingham.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#gives the missing values count in each column
dataset.isna().sum()

#gives the total missing numbers in dataset
dataset.isnull().values.sum()

#replcaing Missing value by mean strategy
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
x=imputer.fit_transform(x)

dataset.isnull().values.sum()
dataset.isna().sum()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,random_state=10)

#Feature Scaling for Improving model Performance
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.fit_transform(X_test)

#Training the logistic regression model (fitting traning datset into model)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='saga', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
classifier.fit(X_train,Y_train)

#Predicting test set results
y_pred=classifier.predict(X_test)

#Evaluating Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
cm

#Accuracy of Model
from sklearn.metrics import accuracy_score
ac=accuracy_score(Y_test,y_pred)
ac