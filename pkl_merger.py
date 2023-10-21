"""
Article: Follow the leader: Index tracking with factor models

Topic: ETC
"""
import os
import pickle
import pandas as pd


def pkl_merge(path: str, file_name: str, keyword: str = 'matched'):
    """
    <DESCRIPTION>
    Merge pickle files into one pickle file.
    """
    directory_path = path
    merged_data = pd.DataFrame()

    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl') and keyword in filename:
            file_path = os.path.join(directory_path, filename)

            data = pd.read_pickle(file_path)

            merged_data = pd.concat([merged_data, data], axis=0)

    with open('{}.pkl'.format(file_name), 'wb') as merged_file:
        pickle.dump(merged_data.reset_index(drop=True), merged_file)
