import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.read_csv('data/climb_stats.csv')
print(data1.columns)
plt.figure()
sns.histplot(data=data1, x='angle')
plt.show()
plt.figure()
sns.histplot(data=data1, x='display_difficulty')
plt.show()

data2 = pd.read_csv('data/climbs.csv')

def count_frames(row):
    frames = row['frames']
    count = frames.count('p')
    return count

data2['length'] = data2.apply(lambda row: count_frames(row), axis=1)
print(data2.head())
print(data2.length)
plt.figure()
sns.histplot(data=data2, x='length')
plt.show()

data3 = pd.read_csv('data/ReducedData.csv')
plt.figure()
sns.histplot(data=data3, x='length')
plt.show()