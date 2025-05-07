import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

data = pd.read_csv('data/Data.csv')

data.drop(data[data['hold_0_x'] == 0].index, inplace = True)
data.dropna(inplace = True)
data.drop(data[data['ascensions'] < 5].index, inplace = True)
print(data.shape)

median_difficulty = data['difficulty'].median()
print(median_difficulty)

data['hard'] = data['difficulty'] < median_difficulty
print(data.columns)

small_data = data[1:2000]

X_train, X_test, Y_train, Y_test = train_test_split(small_data.drop(columns = {'ID', 'difficulty', 'ascensions', 'length', 'hard'}), small_data['hard'], test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_train)

for C in [0.01, 0.1, 1, 10, 100]:
    svc = SVC(kernel='linear', C=C)
    svc.fit(X_train, Y_train)
    scores = cross_val_score(svc, X_test, Y_test, cv = 5)
    print(f"C: {C}, Cross-validation score: {scores.mean()}")