import pandas as pd
import keras as kr
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/ReducedData.csv')

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = {'ID', 'difficulty', 'ascensions', 'length'}), data['difficulty'], test_size = 0.2, random_state = 42)

model = kr.Sequential()
l1 = model.add(kr.layers.Dense(81, activation='relu'))
l2 = model.add(kr.layers.Dense(160, activation='relu'))
l3 = model.add(kr.layers.Dense(60, activation='relu'))
l4 = model.add(kr.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=kr.optimizers.SGD(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2000, batch_size=1000, validation_data=(X_test, y_test))