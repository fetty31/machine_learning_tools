import pandas as pd
import numpy as np
import re

climbs = pd.read_csv('data/climbs.csv')
climb_stats = pd.read_csv('data/climb_stats.csv')
holes = pd.read_csv('data/holes.csv')
placements = pd.read_csv('data/placements.csv')

#Rename climb_uuid to uuid
climb_stats.rename(columns = {"climb_uuid": "uuid"}, inplace=True)
#Rename id to hold_id and hole_id to id
placements.rename(columns={"id": "hold_id", "hole_id": "id"}, inplace=True)
#Delete unnecesary columns
placements.drop(columns={'layout_id', 'set_id', 'default_placement_role_id'}, inplace = True)

#Replace id in holes with real ids from placements
for index, row in holes.iterrows():
  realid = placements.loc[placements['id'] == row["id"]]['hold_id']
  holes.at[index, 'id'] = realid.iloc[0]

#Delete climbs that are not on layout 1 (kilter)
climbs_layout1 = climbs.drop(climbs[climbs.layout_id != 1].index)
#Delete unnecesary columns
climbs_layout1_useful = climbs_layout1.drop(columns = {'setter_id', 'layout_id', 'setter_username', 'name', 'description', 'hsm', 'edge_left', 'edge_right', 'edge_bottom', 'edge_top', 'angle', 'frames_count', 'frames_pace', 'is_draft', 'is_listed', 'created_at'})
#Merge climb data
climbs_layout1_useful_extra = pd.merge(climbs_layout1_useful, climb_stats, on = ['uuid']).drop_duplicates(ignore_index = True)
#Delete unnecesary columns
climbs_data = climbs_layout1_useful_extra.drop(columns = {'benchmark_difficulty', 'quality_average', 'fa_username', 'fa_at'})
#Rename uuid to ID ascensionist_count to number_of_ascensions
climbs_data.rename(columns = {"uuid": "ID", "ascensionist_count": "ascensions", "display_difficulty": "difficulty"}, inplace = True)
#Delete some more
climbs_data.drop(columns = {'difficulty_average'}, inplace = True)
#Swap places
climbs_data=climbs_data.reindex(columns=['ID', 'difficulty', 'ascensions', 'angle', 'frames'])
#Drop climbs with less than 5 ascents
climbs_data.drop(climbs_data[climbs_data['ascensions'] < 5].index, inplace = True)

print(climbs_data.shape)
print(climbs_data.columns)

climbs_data['length'] = climbs_data['frames'].str.count('p')
maxlen = climbs_data.length.max()
maxlen = min(maxlen, 40)
print(maxlen)
climbs_data.drop(climbs_data[climbs_data.length > maxlen].index, inplace = True)
print(climbs_data.shape)


def transform_frames(row, hold_data, max_entries = 40):
    # Split the frames string into individual components
    frame_entries = re.findall(r'p(\d+)r(\d+)', row['frames'])

    # Initialize output lists
    output = []
    start_counter = 0
    finish_counter = 0

    for id_str, type_str in frame_entries:

        hold_data = holes.loc[holes['id'] == int(id_str)]
        position = [int(hold_data.x.iloc[0]), int(hold_data.y.iloc[0])]

        # Convert type to binary flags
        type_int = int(type_str)
        if type_int == 12:
            flags = [1, 0, 0]
        elif type_int == 14:
            flags = [0, 0, 1]
        elif type_int == 15:
            flags = [0, 1, 0]
        else:
            flags = [0, 0, 0]  # Default for unknown types

        # Combine all values for this entry
        output.extend([*position, *flags])

    print(row.name)

    if start_counter > 2 or finish_counter > 2:
        output = [0]

    # Zero-pad to reach max_entries * 5 columns (2 lookup + 3 flags per entry)
    padded_output = output + [0] * (max_entries * 5 - len(output))
    return padded_output


def process_dataframe(climb_data, hold_data, maxlen = 40):
    # Apply the transformation to each row
    data = climb_data.apply(lambda row: transform_frames(row, hold_data, maxlen), axis=1)

    # Create column names
    columns = []
    for i in range(40):  # For max 40 entries
        columns.extend([
            f'hold_{i}_x',
            f'hold_{i}_y',
            f'hold_{i}_is_start',
            f'hold_{i}_is_foothold',
            f'hold_{i}_is_finish'
        ])

    # Create new dataframe with transformed data
    transformed_df = pd.DataFrame(data.tolist(), columns=columns)

    print(transformed_df.shape)
    print(climb_data.shape)

    # Combine with original dataframe (excluding frames column)
    result_df = pd.concat([
        climb_data.drop(columns=['frames']).reset_index(drop=True),
        transformed_df.reset_index(drop=True)
    ], axis = 1)

    return result_df

data = process_dataframe(climbs_data, holes, maxlen)

print(data.shape)
print(data.columns)
print(data.head())

data.drop(data[data['hold_0_x'] == 0].index, inplace = True)
data.dropna(inplace = True)

print(data.shape)
print(data.columns)
print(data.head())

data.to_csv('data/Data.csv', index = False)
