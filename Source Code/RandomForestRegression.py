import numpy as np
import pandas as pd
import math as mp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Gathering and setting the data.
data = pd.read_csv("Ce475ProjectData.csv")
x = data.iloc[:100, 1: 7].values
y = data.iloc[:100, 7].values
x_predict = data.iloc[100:120, 1: 7].values

# In order to do the partitioning on the train and test data, I have imported the train_test_split library from sklearn
# Split arrays or matrices into random train and test subsets

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)


regression = RandomForestRegressor(n_estimators = 1000, random_state = 0 )
regression.fit(X_train, Y_train)
Y_pred = regression.predict(X_test)


mse = mean_squared_error(Y_pred, Y_test)

mp.sqrt(mse)

accuracies = cross_val_score(regression, X_train,  Y_train, cv = 10)
accuracies.mean()
accuracies.std()

y_new_pred = regression.predict(x_predict)


r2_score(Y_test, Y_pred)