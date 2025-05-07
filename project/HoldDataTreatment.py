import pandas as pd

data = pd.read_csv('data/Data.csv')
print(data.shape)

def process_holds(df, maxholds = 40):

    holds = []

    for i in range(maxholds):
        x = f'hold_{i}_x'
        y = f'hold_{i}_y'
        f1 = f'hold_{i}_is_start'
        f2 = f'hold_{i}_is_foothold'
        f3 = f'hold_{i}_is_finish'

        if x not in df.columns:
            continue

        # Create a temporary DataFrame for this entry
        temp_df = df[[x, y, f1, f2, f3]].copy()
        temp_df.columns = ['hold_x', 'hold_y', 'hold_is_start', 'hold_is_foothold', 'hold_is_finish']

        # Filter out all-zero rows
        temp_df = temp_df[(temp_df != 0).any(axis=1)]

        holds.append(temp_df)

    # Combine all entries into one DataFrame
    if holds:
        return pd.concat(holds, ignore_index=True)
    else:
        return pd.DataFrame(columns=['hold_x', 'hold_y', 'hold_is_start', 'hold_is_foothold', 'hold_is_finish'])

hold_data = process_holds(data)
print(hold_data.shape)
print(hold_data.columns)

hold_data.to_csv('data/hold_data.csv', index = False)