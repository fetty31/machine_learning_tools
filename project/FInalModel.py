import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

#Data import and split:
data = pd.read_csv('project/ReducedData20.csv')
X = data.drop(columns=['ID', 'length', 'difficulty', 'ascensions'])
Y = data['difficulty']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)

print(f"X_train shape: {np.shape(X_train)}")
print(f"X_test shape: {np.shape(X_test)}")
print(f"Y_train shape: {np.shape(Y_train)}")
print(f"Y_test shape: {np.shape(Y_test)}")

#Results:

# HistGradientBoosting: (best)
model = HistGradientBoostingRegressor(random_state=10, max_iter=1000, learning_rate=0.1, max_depth=8)
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
Y_pred = model.predict(X_test)

print('--------- Done with HistGradientBoosting Regressor')
print(f"R2 score: {score}")

plt.figure(figsize=(7,7))
plt.scatter(Y_pred, Y_test)
plt.title("Predicted vs Real")
plt.show()