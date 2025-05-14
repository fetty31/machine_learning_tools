import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/ReducedData.csv')
print(data.shape)

X = data.drop(columns=['ID', 'length', 'difficulty', 'ascensions'])
Y = data['difficulty']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

regressor = AdaBoostRegressor(n_estimators=100, random_state=10)
regressor.fit(X_train, Y_train)
print(regressor.score(X_test, Y_test))