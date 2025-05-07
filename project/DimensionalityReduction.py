import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

hold_data = pd.read_csv('data/hold_data.csv')

pca = PCA(n_components=2)
pca.fit(hold_data)

data = pd.read_csv('data/Data.csv')

def reduce_dimension(row, pca, maxholds = 40):

    temp_df = pd.DataFrame(columns = ['hold_x', 'hold_y', 'hold_is_start', 'hold_is_foothold', 'hold_is_finish'])

    for i in range(maxholds):
        x = f'hold_{i}_x'
        y = f'hold_{i}_y'
        f1 = f'hold_{i}_is_start'
        f2 = f'hold_{i}_is_foothold'
        f3 = f'hold_{i}_is_finish'

        temp_df.loc[i]  = row[[x, y, f1, f2, f3]].tolist()

    reduced_data = pca.transform(temp_df).tolist()
    reduced_data = np.reshape(reduced_data, 2*maxholds)
    print(row.name)

    return reduced_data

def apply_pca(data, pca, maxholds = 40):
    # Apply the transformation to each row
    data_reduced = data.apply(lambda row: reduce_dimension(row, pca, maxholds), axis=1)

    # Create column names
    columns = []
    for i in range(maxholds):
        columns.extend([
            f'hold_{i}_feature1',
            f'hold_{i}_feature2',
        ])

    # Create new dataframe with transformed data
    transformed_df = pd.DataFrame(data_reduced.tolist(), columns=columns)

    # Combine with original dataframe (excluding frames column)
    result_df = pd.concat([
        data[['ID', 'difficulty', 'ascensions', 'angle', 'length']].reset_index(drop=True),
        transformed_df.reset_index(drop=True)
    ], axis=1)

    return result_df

reduced_data = apply_pca(data, pca, 40)

print(reduced_data.shape)
print(reduced_data.columns)
print(reduced_data.head())

reduced_data.to_csv('data/ReducedData.csv', index = False)