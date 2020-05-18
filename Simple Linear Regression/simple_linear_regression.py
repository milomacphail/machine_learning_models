# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:29:53 2020
@author: milom
Simple Linear Regression
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Simple Linear Regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predictor
y_pred = regressor.predict(X_test)

#Visualization of data points
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title("Expected salary by years of experience(training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Visualization of data points
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title("Expected salary by years of experience(test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


