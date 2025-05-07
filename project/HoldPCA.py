import pandas as pd
from sklearn.decomposition import PCA
import pickle as pk

data = pd.read_csv('data/hold_data.csv')

pca = PCA(n_components=2)
pca.fit(data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)

pk.dump(pca, open('data/hold_pca.pkl', 'wb'))