import pandas as pd
from sklearn.neighbors import NearestNeighbors

def NN(data, numberofneighbors, treat_col):
    control_group = data[data[treat_col] == 0].copy()
    treatment_group = data[data[treat_col] == 1].copy()

    nn = NearestNeighbors(n_neighbors=len(control_group), metric='euclidean')
    nn.fit(control_group[['pscore']])
    distances, indices = nn.kneighbors(treatment_group[['pscore']]) # Finding nearst control unit

    # Track used control indices
    used_control_indices = set()
    matched_pairs = []

    # Loop through each treated observation and find multiple closest available control units
    for i, treated_index in enumerate(treatment_group.index):
        count = 0  # Track how many matches this treated unit has received

        for neighbor_idx in range(len(control_group)):  # Iterate through potential matches
            control_index = control_group.index[indices[i, neighbor_idx]]

            if control_index not in used_control_indices:  # Ensure no replacement
                used_control_indices.add(control_index)  # Mark control as used
                matched_pairs.append({
                    'treated_index': treated_index,
                    'control_index': control_index,
                    'distance': distances[i, neighbor_idx]
                })
                count += 1

            if count >= numberofneighbors:  # Stop after reaching the required number of matches
                break

    # Convert to DataFrame
    matched_df = pd.DataFrame(matched_pairs)

    # Extract matched observations
    matched_treated = treatment_group.loc[matched_df['treated_index']].reset_index(drop=True)
    matched_controls = control_group.loc[matched_df['control_index']].reset_index(drop=True)
    # Combine matched pairs
    matched_data = pd.concat([matched_treated, matched_controls], axis=0).reset_index(drop=True)
    matched_data = matched_data.drop_duplicates(subset=['ID'], keep='last')
    # matched_data = matched_data.merge(data[['nsmoid', 'pscore']], on='nsmoid', how='left')

    return matched_data
