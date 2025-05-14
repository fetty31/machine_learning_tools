import pandas as pd
import keras as kr
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/ReducedData.csv')

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = {'ID', 'difficulty', 'ascensions', 'length'}), data['difficulty'], test_size = 0.2, random_state = 42)

model = kr.Sequential()
l1 = model.add(kr.layers.Dense(240, input_dim=81, activation='relu'))
l2 = model.add(kr.layers.Dense(240, activation='relu'))
l3 = model.add(kr.layers.Dense(160, activation='relu'))
l4 = model.add(kr.layers.Dense(160, activation='relu'))
l5 = model.add(kr.layers.Dense(80, activation='relu'))
l6 = model.add(kr.layers.Dense(20, activation='relu'))
l7 = model.add(kr.layers.Dense(10, activation='relu'))
l8 = model.add(kr.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['r2_score'])

model.fit(X_train, y_train, epochs=100, batch_size = 10, validation_data=(X_test, y_test))