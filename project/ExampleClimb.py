import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor

#Data import and split:
data = pd.read_csv('project/ReducedData20.csv')
X = data.drop(columns=['ID', 'length', 'difficulty', 'ascensions'])
Y = data['difficulty']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)

print(f"X_train shape: {np.shape(X_train)}")
print(f"X_test shape: {np.shape(X_test)}")
print(f"Y_train shape: {np.shape(Y_train)}")
print(f"Y_test shape: {np.shape(Y_test)}")

X_id = data[data["ID"] == "bf0db71d0d674289beef46e226216bf6"] # pick example climb
X_ = X_id[X_id["angle"] == 40]
print(X_)
X_test = X_.drop(columns=['ID', 'length', 'difficulty', 'ascensions'])

#Results:

# HistGradientBoosting: (best)
model = HistGradientBoostingRegressor(random_state=10, max_iter=1000, learning_rate=0.1, max_depth=8)
model.fit(X_train, Y_train)
# # score = model.score(X_test, Y_test)
Y_pred = model.predict(X_test)
print(f"HistGradientBoosting: {Y_pred}")

linear = LinearRegression()
linear.fit(X_train, Y_train)
Y_pred = linear.predict(X_test)
print(f"LinearRegression: {Y_pred}")

decisiontree = DecisionTreeRegressor(random_state=10, max_depth=10)
decisiontree.fit(X_train, Y_train)
Y_pred = decisiontree.predict(X_test)
print(f"DecisionTree: {Y_pred}")

randomforest = RandomForestRegressor(random_state=10, max_depth=10, n_estimators=1000)
randomforest.fit(X_train, Y_train)
Y_pred = randomforest.predict(X_test)
print(f"Randomforest: {Y_pred}")

extratree = ExtraTreesRegressor(random_state=10, max_depth=10, min_samples_leaf=1)
extratree.fit(X_train, Y_train)
Y_pred = extratree.predict(X_test)
print(f"Extratree: {Y_pred}")

gradientboosting = GradientBoostingRegressor(random_state=10, learning_rate=0.1, n_estimators=1000)
gradientboosting.fit(X_train, Y_train)
Y_pred = gradientboosting.predict(X_test)
print(f"GradientBoosting: {Y_pred}")