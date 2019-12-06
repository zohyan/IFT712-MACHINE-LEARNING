import os
import pandas as pd


class DataLoader:

    def __init__(self):
        self.path = 'datasets'
        self.train_data, self.test_data = None, None

    def load_data(self):
        """
        :return: Return the three raw datasets
        """
        self.train_data = pd.read_csv(os.path.join(self.path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(self.path, 'test.csv'))
        gender_submission = pd.read_csv(os.path.join(self.path, 'gender_submission.csv'))

        return self.train_data, self.test_data, gender_submission