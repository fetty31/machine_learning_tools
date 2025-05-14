import pandas as pd

data = pd.read_csv('data/ReducedData.csv')
print(data.shape)
data.drop(data[data['length'] > 20].index, inplace=True)
print(data.shape)
cols = []
for i in range(20, 40):
    c1 = 'hold_'+str(i)+'_feature1'
    c2 = 'hold_'+str(i)+'_feature2'
    cols.append(c1)
    cols.append(c2)
data.drop(columns=cols, inplace=True)
data.to_csv('data/ReducedData.csv', index=False)
