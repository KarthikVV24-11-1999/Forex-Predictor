#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:45:18 2019

@author: venkatakarthikvadlamudi
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
from sklearn.linear_model import LinearRegression

# Read the data in the CSV file using pandas
# Change the path to the .csv file accordingly!
dataframe = pd.read_csv('train.csv')
testframe = pd.read_csv('test.csv')
# Filling NaN with mean in training dataframe
dataframe .isnull().any().sum()
dataframe.fillna(-0.2,inplace=True)
# Filling NaN with mean in testing dataframe
testframe .isnull().any().sum()
testframe.fillna(testframe.mean(),inplace=True)
# Creating a dataframe to store target of training dataframe
target = pd.DataFrame(dataframe['target'])
# Creating a dataframe to store id of training dataframe
unique_id = pd.DataFrame(dataframe['id'])
# Creating a dataframe to store span of training dataframe
span = dataframe['span']
# Extracting only features by dropping target and id from training dataframe
features = dataframe.drop(['target','id'], axis = 1)
features_array = features.values
# Extracting only target of training dataframe
target_array = target.values
X_train = features_array
y_train = target_array
# Extracting only features by dropping id from testing dataframe
X_test = testframe.drop(['id'], axis = 1)
X_test = X_test.values

# Standard scaler for scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Defining classifer, here we have used RBF kernel. We could change the values of C and the kernel function and see the results
classifier = svm.SVC(C=4,kernel='sigmoid',gamma = "scale")
#classifier = LinearRegression()
# Encoding labels (this is required as the code was cribbing otherwise. There seems to be string data somewhere)
lab_enc = LabelEncoder()
y_train = y_train.ravel()
y_train = lab_enc.fit_transform(y_train)
print("TRAINING START")
# Fitting the curve (learning)
classifier.fit(X_train,y_train)
# Predicting the target (testing)
print("TESTING START")
predicted = classifier.predict(X_test)
predicted = -pd.DataFrame(lab_enc.fit_transform(predicted))
# Scaling the predicted target
mmsc = MinMaxScaler(feature_range=(-1, 1))
predicted = mmsc.fit_transform(predicted)
predicted = mmsc.transform(predicted)
output=np.array(predicted)
pd.DataFrame(output).to_csv("file5.csv")
print("Done")
# Check the values of predicted which are the predicted target values. We cannot validate the predicted target values.