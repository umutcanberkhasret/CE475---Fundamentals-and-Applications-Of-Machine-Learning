import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn import linear_model


# Gathering and setting the data.
data = pd.read_csv("Ce475ProjectData.csv")
x = data.iloc[:100, 1: 7].values
y = data.iloc[:100, 7].values

# In order to do the partitioning on the train and test data, I have imported the train_test_split library from sklearn
# Split arrays or matrices into random train and test subsets

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# =============================================================================
# To convert them into Nx1 rather than 1xN
# Y_train = Y_train.reshape(-1, 1)
# Y_test = Y_test.reshape(-1, 1)
# 
#  We are implementing both linear and polynomial regressions to see if there exists a difference among them in terms
#  of performance and accuracy.
# =============================================================================

lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)

# Building Polynomial Regression; After trying 2nd 3rd and 4th degrees,
# it would be fair to say we have obtained the best result in degree 2.
poly_regression = PolynomialFeatures(degree= 2) 
# =============================================================================
#   as we increase the degree, the overall r^2 value decreases which results in
#   way worse estimations
# =============================================================================
X_poly = poly_regression.fit_transform(X_train)
poly_regression.fit(X_poly, Y_train)
linear_regression = LinearRegression()
linear_regression.fit(X_poly, Y_train)

y_prediction = linear_regression.predict(poly_regression.fit_transform(X_test))


# SCORING THE MODEL

r2_score(Y_test, y_prediction)

# =============================================================================
# applying Lasso Regression to decrease the model's complexity by making the coefficients that
# are unnecessary either 0 or close to 0.
# 
# to see the effect that is done by using Lasso Regression, checked both of the cases.
# =============================================================================

lasso = linear_model.Lasso()
accuracy = cross_val_score(linear_regression, X_train, Y_train, cv = 3)
accuracy.mean()
accuracy.std()

accuracy3 = cross_val_score(lasso,X_train, Y_train, cv = 3)
accuracy3.mean()
accuracy3.std()




# =============================================================================
# # r^2 values are still to low, however there is a slight improvement comparing with
# # multiple linear regression
# 
# =============================================================================
