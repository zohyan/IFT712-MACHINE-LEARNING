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

    def completing_age_features(self):
        """
        :return: The dataset in which the nulls values in the Age features have been completed
        """
        median_age = np.zeros((2, 3))

        for row in self.dataset:
            for sex in range(0, 2):
                for pclass in range(0, 3):
                    guess_df = row[(row['Sex'] == sex) & (row['Pclass'] == pclass + 1)]['Age'].dropna()
                    age_guess = guess_df.median()
                    median_age[sex, pclass] = age_guess

        for row in self.dataset:
            for i in range(0, 2):
                for j in range(0, 3):
                    row.loc[(row.Age.isnull()) & (row.Sex == i) & (row.Pclass == j + 1), 'Age'] = median_age[i, j]

            row['Age'] = row['Age'].astype(int)

        return self.dataset

    def change_categorical_feature_sex_to_numerical(self, df_train, df_test):
        """
        df_train: The train dataframe
        df_test: The test dataframe
        :return: The categorical sex variable in each dataframe in numerical form
        """
        df_train.loc[df_train['Sex'] == 'male', 'Sex'] = 1
        df_train.loc[df_train['Sex'] == 'female', 'Sex'] = 0

        df_test.loc[df_test['Sex'] == 'male', 'Sex'] = 1
        df_test.loc[df_test['Sex'] == 'female', 'Sex'] = 0

        return df_train, df_test

    def completing_embarked_features(self):
        """
        :return: The dataset in which the nulls values in the Embarkes features have been completed
        """
        freq_port = self.train_data.Embarked.dropna().mode()[0]

        for row in self.dataset:
            row['Embarked'] = row['Embarked'].fillna(freq_port)

        return self.dataset

    def change_categorical_feature_embarked_to_numerical(self):
        """
        :return: The categorical Embarked variable in the dataset is convert to numerical form
        """
        for row in self.dataset:
            row['Embarked'] = row['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        return self.dataset

    def completing_fare_features(self, df_test):
        """
        :return: The dataset in which the nulls values in the Fare features have been completed
        """
        df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)

    def naive_preprocessing_data(self):
        """
            This method is written just to have a data that will be classifiable to have a benchmark.
            We will do so :
            - Deletion of features that seems to be irrelevant
            - Converting a categorical feature to numerical feature
            - And completing missing value
        """

        self.train_data, self.test_data, gender_submission = self.load_data()

        # Here we delete features
        features_to_delete = ['PassengerId', 'Ticket', 'Cabin', 'Name']

        self.train_data = self.delete_features(self.train_data, features_to_delete)
        self.test_data = self.delete_features(self.test_data, features_to_delete)

        # We affect the result to self.dataset
        self.dataset = [self.train_data, self.test_data]

        # Here we convert categorical features Sex to numerical
        self.train_data, self.test_data = self.change_categorical_feature_sex_to_numerical(self.train_data,
                                                                                           self.test_data)

        # Complete null values in the Age features
        self.completing_age_features()

        # Complete null values in the Embarked features
        self.completing_embarked_features()

        # Here we convert categorical features Embarked to numerical
        self.change_categorical_feature_embarked_to_numerical()

        # Complete null values in the Fare features
        self.completing_fare_features(self.test_data)

        # Get the data
        x_train = self.delete_features(self.train_data, ['Survived'])
        y_train = self.train_data["Survived"]
        x_test = self.test_data
        y_test = gender_submission["Survived"]

        return x_train, y_train, x_test, y_test