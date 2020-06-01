import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.formula.api as sm
from sklearn.metrics import r2_score


# Gathering and setting the data.
data = pd.read_csv("Ce475ProjectData.csv")
x = data.iloc[:100, 1: 7].values
y = data.iloc[:100, 7].values

# In order to do the partitioning on the train and test data, I have imported the train_test_split library from sklearn
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Here we will fit the model to our training set
regression = LinearRegression()
regression.fit(X_train, Y_train) # train set is placed

# We'll predict the test set that we have separated before, the consequences are too distinct, there are probably
# variables exists that has nothing to do with the dependent variable.
y_prediction = regression.predict(X_test)

# Implementation of the backward Elimination method to increase the model's accuracy.
X_temp = np.append(arr= np.ones((80, 1)).astype(int), values= X_train, axis=1)
X_optimal = X_temp[:, [0, 1, 2, 3, 4, 5, 6]]

# =============================================================================
# # Ordinary Least Squares
# # regression_OLS.summary function is used to display which variable to eliminate at
# # every step of backward elimination
# =============================================================================
regression_OLS = sm.OLS(endog=Y_train, exog= X_optimal).fit()

print regression_OLS.summary()

# The P value of x5 was above our confidence level p = 0.05. So, we'll remove it
X_optimal = X_temp[:, [0, 1, 2, 3, 4, 6]]
# Ordinary Least Squares
regression_OLS = sm.OLS(endog=Y_train, exog= X_optimal).fit()
print regression_OLS.summary()

X_optimaltest = X_test[:, [0,1,2,3,5]]
X_optimaltest = np.append(arr =  np.ones((20, 1)).astype(int), values = X_test, axis = 1)

# =============================================================================
# #after the step above, the data integration is corrupted, unwanted x5 column is also 
# # comes back to the consideration, that's why we exec the following statement
# =============================================================================

X_optimaltest = X_optimaltest[:, [0,1,2,3,4,6]]

linear_reg = LinearRegression()
linear_reg.fit(X_optimal, Y_train)

y_new_prediction = linear_reg.predict(X_optimaltest)

# SCORING THE ACCURACY


accuracy = cross_val_score(regression, X_train, Y_train, cv = 5)
accuracy.mean()
accuracy.std()

# r^2 calculation to see what is the success when we apply backward elimination
r2_score(Y_test, y_new_prediction)

# r^2 calculation again, the difference here, the model that was used here doesn't processed with
# backward elimination. Every variable is used for the calculation
r2_score(Y_test, y_prediction)

# R^2 values are prett low, the values are minus



