import pandas as pd
import numpy as np
import os

class DataPreprocessing:

    def __init__(self):
        self.path = 'datasets'
        self.train_data, self.test_data, self.dataset = None, None, None

    def load_data(self):
        """
        :return: Return the three raw datasets
        """
        self.train_data = pd.read_csv(os.path.join(self.path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(self.path, 'test.csv'))
        gender_submission = pd.read_csv(os.path.join(self.path, 'gender_submission.csv'))

        return self.train_data, self.test_data, gender_submission

    def delete_features(self, df, features_to_delete):
        """
        df: The dataframe that contains the features to delete
        features_to_delete: list of features to delete
        :return: The dataframe that without the features that have been deleted
        """
        return df.drop(features_to_delete, axis=1)