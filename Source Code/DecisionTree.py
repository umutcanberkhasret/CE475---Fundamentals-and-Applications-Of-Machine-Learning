import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# Gathering and setting the data.
data = pd.read_csv("Ce475ProjectData.csv")
x = data.iloc[:100, 1: 7].values
y = data.iloc[:100, 7].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
regression = DecisionTreeRegressor(random_state=0)
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

# obtaining the r^2 value to measure the success of the algorithm
r2_score(y_test, y_pred)

accuracy = cross_val_score(regression, x_train, y_train, cv = 5)
accuracy.mean()
accuracy.std()

# There is a huge increase about the success when we use decision trees in 
# r2 values. They are nearly 0.7 which means decision tree algorithm provides
# a better insight and increases the quality of estimations