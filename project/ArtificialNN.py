import pandas as pd
import keras as kr
import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/ReducedData.csv')

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = {'ID', 'difficulty', 'ascensions', 'length'}), data['difficulty'], test_size = 0.2, random_state = 42)

model = kr.Sequential()
l1 = model.add(kr.layers.Dense(240, input_dim=41, activation='relu'))
l2 = model.add(kr.layers.Dense(240, activation='relu'))
l3 = model.add(kr.layers.Dropout(0.02, noise_shape=None))
l4 = model.add(kr.layers.Dense(160, activation='relu'))
l5 = model.add(kr.layers.Dense(160, activation='relu'))
l6 = model.add(kr.layers.Dropout(0.02, noise_shape=None))
l7 = model.add(kr.layers.Dense(80, activation='relu'))
l8 = model.add(kr.layers.Dense(40, activation='relu'))
l9 = model.add(kr.layers.Dense(20, activation='relu'))
l10 = model.add(kr.layers.Dense(10, activation='relu'))
l11 = model.add(kr.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['r2_score'])

cb = kr.callbacks.EarlyStopping(monitor='val_r2_score', patience=8, restore_best_weights=True)

for batch_size in [100, 300, 800, 1000, 3000, 8000, 10000]:
    model.fit(X_train, y_train, batch_size=batch_size, epochs=100, callbacks=[cb], validation_data=(X_test, y_test))