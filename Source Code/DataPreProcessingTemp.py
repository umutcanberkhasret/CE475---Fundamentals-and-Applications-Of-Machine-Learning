"""
This file is implemented to decrease the amount of time that will be spent with data preparation
while we are implementing the next algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv("Ce475ProjectData.csv")

# x1 = data.iloc[:100,1].values.reshape(-1,1)
# x2 = data.iloc[:100,2].values.reshape(-1,1)
# x3 = data.iloc[:100,3].values.reshape(-1,1)
# x4 = data.iloc[:100,4].values.reshape(-1,1)
# x5 = data.iloc[:100,5].values.reshape(-1,1)
# x6 = data.iloc[:100,6].values.reshape(-1,1)

x = data.iloc[:100, 1: 7].values.astype(float)
# used reshape function to make the array 100x1. It was 1x100 which was ending up with dimension error
y = data.iloc[:100, 7].values.reshape(-1, 1).astype(float)

x_predict = data.iloc[100:, 1: 7].values.astype(float)
# In order to do the partitioning on the train and test data, I have imported the train_test_split library from sklearn
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""
#                                               Can be used if needed; Feature Scaling
# Doing feature scaling here for two reason;
# -> First is to increase the performance of the algorithm,
# -> Second, since the range of the variables is huge, to prevent the domination between them we need to convert them
#    into same range to reduce the difference that would happen when we take their squares and differences
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Y_train = sc_X.fit_transform(Y_train)
Y_test = sc_X.transform(Y_test)
"""

