import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/ReducedData.csv')
print(data.shape)

X = data.drop(columns=['ID', 'length', 'difficulty', 'ascensions'])
Y = data['difficulty']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

regressor = MLPRegressor(max_iter=20000, random_state=10, tol=0.01)
regressor.fit(X_train, Y_train)
print(regressor.score(X_test, Y_test))