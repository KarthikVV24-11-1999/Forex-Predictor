#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:08:20 2019
@author: saianuroopkesanapalli
"""

# import numpy, pandas and time
import numpy as np
import pandas as pd

# visual libraries
from matplotlib import pyplot as plt
plt.style.use('ggplot')

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

# Read the data in the CSV file using pandas
# Change the path to the .csv file accordingly!
dataframe = pd.read_csv('/Users/saianuroopkesanapalli/Desktop/5th Semester/BitGrit Challenge/competition_data/train.csv')
# Filling NaN with mean
dataframe .isnull().any().sum()
dataframe.fillna(dataframe.mean(),inplace=True)
# Creating a dataframe to store target
target = pd.DataFrame(dataframe['target'])
# Creating a dataframe to store id
unique_id = pd.DataFrame(dataframe['id'])
# Creating a dataframe to store span
span = dataframe['span']
# Plotting target vs id
dataframe.plot(x = 'id',y = 'target')
# Plotting span vs id
dataframe.plot(x = 'id',y = 'span')
# Extracting only features by dropping target and id
features = dataframe.drop(['target','id'], axis = 1)
features_array = features.values
# Extracting only target
target_array = target.values
# Splitting train set into 80:20 for training and testing
# We are actually supposed to train on this set and test on the other given set. We will implement this soon as this is just a skeletal code
X_train,X_test,y_train,y_test = train_test_split(features_array,target_array,test_size=0.20)
# Standard scaler for scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Defining classifer, here we have used RBF kernel. We could change the values of C and the kernel function and see the results
classifier = svm.SVC(C=1,kernel='rbf',gamma = "scale")
# Encoding labels (this is required as the code was cribbing otherwise. There seems to be string data somewhere)
lab_enc = LabelEncoder()
y_train = y_train.ravel()
y_train = lab_enc.fit_transform(y_train)
# Fitting the curve (learning)
classifier.fit(X_train,y_train)
# Predicting the target (testing)
predicted = classifier.predict(X_test)
predicted = -pd.DataFrame(lab_enc.fit_transform(predicted))
# Scaling the predicted target
mmsc = MinMaxScaler(feature_range=(-1, 1))
predicted = mmsc.fit_transform(predicted)
predicted = mmsc.transform(predicted)
# Calculating Mean Sqyared Error (this is our evaluation criterion)
score = mean_squared_error(y_test,predicted)
print("MSE:" + str(score))

# Compare y-test and predicted to check the predicted target values by yourself