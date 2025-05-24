import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
import xgboost as xg

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
results = pd.DataFrame(columns=['Regressor', 'R2_Score', 'Parameters'])

#Dummy regressor:
# dummy = DummyRegressor(strategy='mean')
# dummy.fit(X_train, Y_train)
# results.loc[0] = ['DummyRegressor', dummy.score(X_test, Y_test), 0]
# print('--------- Done with Dummy Regressor')
# print(results.loc[0])

# #Linear regressor:
# linear = LinearRegression()
# linear.fit(X_train, Y_train)
# results.loc[1] = ['LinearRegressor', linear.score(X_test, Y_test), 0]
# print('--------- Done with Linear Regressor')
# print(results.loc[1])

# #Decisiontree regressor:
# parameters = {'max_depth': range(6,12,2)}
# decisiontree = DecisionTreeRegressor(random_state=10)
# decisiontreegs = GridSearchCV(estimator=decisiontree, param_grid=parameters, n_jobs=-1)
# decisiontreegs.fit(X_train, Y_train)
# results.loc[2] = ['DecisionTreeRegressor', decisiontreegs.score(X_test, Y_test), decisiontreegs.best_params_]
# print('--------- Done with DecisionTree Regressor')
# print(results.loc[2])

# #Randomforest regressor: (best)
# parameters = {'max_depth': [8, 10], 'n_estimators': np.logspace(2,3,3, dtype=int)}
# randomforest = RandomForestRegressor(random_state=10, n_jobs=-1)
# randomforestgs = GridSearchCV(estimator=randomforest, param_grid=parameters, n_jobs=-1, cv=2)
# randomforestgs.fit(X_train, Y_train)
# results.loc[3] = ['RandomForestRegressor', randomforestgs.score(X_test, Y_test), randomforestgs.best_params_]
# print('--------- Done with RandomForest Regressor')
# print(results.loc[3])

# #Extratree regressor:
# parameters = {'max_depth': [8, 10], 'min_samples_leaf': [1, 2, 3]}
# extratree = ExtraTreesRegressor(random_state=10, n_jobs=-1)
# extratreegs = GridSearchCV(estimator=extratree, param_grid=parameters, n_jobs=-1)
# extratreegs.fit(X_train, Y_train)
# results.loc[4] = ['ExtraTreeRegressor', extratreegs.score(X_test, Y_test), extratreegs.best_params_]
# print('--------- Done with ExtraTree Regressor')
# print(results.loc[4])

#GradientBoosting:
# parameters = {'learning_rate': np.logspace(-2, 0, 3), 'n_estimators': np.logspace(2,3,3, dtype=int)}
# gradientboosting = GradientBoostingRegressor(random_state=10)
# gradientboostinggs = GridSearchCV(estimator=gradientboosting, param_grid=parameters, n_jobs=-1)
# gradientboostinggs.fit(X_train, Y_train)
# results.loc[5] = ['GradientBoostingRegressor', gradientboostinggs.score(X_test, Y_test), gradientboostinggs.best_params_]
# print('--------- Done with GradientBoosting Regressor')
# print(results.loc[5])

# #Adaboost:
# parameters = {'learning_rate': np.logspace(-2, 0, 5), 'n_estimators': np.logspace(2,3,5, dtype=int)}
# adaboost = AdaBoostRegressor(random_state=10)
# adaboostgs = GridSearchCV(estimator=adaboost, param_grid=parameters, n_jobs=-1)
# adaboostgs.fit(X_train, Y_train)
# results.loc[6] = ['AdaBoostRegressor', adaboostgs.score(X_test, Y_test), adaboostgs.best_params_]
# print('--------- Done with AdaBoost Regressor')
# print(results.loc[6])

# HistGradientBoosting: (best)
parameters = {'learning_rate': np.logspace(-2, 0, 5), 'max_iter': np.logspace(2,5,5, dtype=int), 'max_depth': range(6,12,2)}
histgradientboosting = HistGradientBoostingRegressor(random_state=10)
histgradientboostinggs = GridSearchCV(estimator=histgradientboosting, param_grid=parameters, n_jobs=-1)
histgradientboostinggs.fit(X_train, Y_train)
results.loc[7] = ['HistGradientBoostingRegressor', histgradientboostinggs.score(X_test, Y_test), histgradientboostinggs.best_params_]
print('--------- Done with HistGradientBoosting Regressor')
print(results.loc[7])
print(histgradientboostinggs.best_params_)

#XGBoost:
# parameters = {'max_depth': range(612,2), 'eta': np.logspace(-2, 0, 5), 'gamma': np.logspace(0, 1, 3)}
# xgboost = xg.XGBRegressor(tree_method='hist', device='cuda', nthread=-1)
# xgboostgs = GridSearchCV(estimator=xgboost, param_grid=parameters, n_jobs=-1)
# xgboostgs.fit(X_train, Y_train)
# results.loc[8] = ['XGBoost', xgboostgs.score(X_test, Y_test), xgboostgs.best_params_]
# print('--------- Done with XGBoost Regressor')
# print(results.loc[8])

# results.to_csv('project/preliminary_results2.csv', index=False)