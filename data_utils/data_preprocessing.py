import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from .data_loader import DataLoader


class DataPreprocessing:

    def __init__(self):
        self.dataset = None
        self.train_data, self.test_data, self.gender_submission = DataLoader().load_data()

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

        self.train_data['Embarked'] = self.train_data['Embarked'].fillna(freq_port)
        self.test_data['Embarked'] = self.test_data['Embarked'].fillna(freq_port)

        self.dataset = [self.train_data, self.test_data]

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

    # preprocess vaovao
    def creating_title_feature(self):

        for dataset in self.dataset:
            dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

        for dataset in self.dataset:
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                        'Rare')

            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        for dataset in self.dataset:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)

    def create_age_band_features(self):
        self.train_data.loc[self.train_data['Age'] <= 16, 'Age'] = 0
        self.train_data.loc[(self.train_data['Age'] > 16) & (self.train_data['Age'] <= 32), 'Age'] = 1
        self.train_data.loc[(self.train_data['Age'] > 32) & (self.train_data['Age'] <= 48), 'Age'] = 2
        self.train_data.loc[(self.train_data['Age'] > 48) & (self.train_data['Age'] <= 64), 'Age'] = 3
        self.train_data.loc[self.train_data['Age'] > 64, 'Age']

        self.test_data.loc[self.test_data['Age'] <= 16, 'Age'] = 0
        self.test_data.loc[(self.test_data['Age'] > 16) & (self.test_data['Age'] <= 32), 'Age'] = 1
        self.test_data.loc[(self.test_data['Age'] > 32) & (self.test_data['Age'] <= 48), 'Age'] = 2
        self.test_data.loc[(self.test_data['Age'] > 48) & (self.test_data['Age'] <= 64), 'Age'] = 3
        self.test_data.loc[self.test_data['Age'] > 64, 'Age']

        self.dataset = [self.train_data, self.test_data]

    def create_is_alone_feature(self):

        self.train_data['FamilySize'] = self.train_data['SibSp'] + self.train_data['Parch'] + 1

        self.train_data['IsAlone'] = 0
        self.train_data.loc[self.train_data['FamilySize'] == 1, 'IsAlone'] = 1

        self.test_data['FamilySize'] = self.test_data['SibSp'] +self.test_data['Parch'] + 1

        self.test_data['IsAlone'] = 0
        self.test_data.loc[self.test_data['FamilySize'] == 1, 'IsAlone'] = 1

    def create_fare_band(self):
        self.train_data.loc[self.train_data['Fare'] <= 7.91, 'Fare'] = 0
        self.train_data.loc[(self.train_data['Fare'] > 7.91) & (self.train_data['Fare'] <= 14.454), 'Fare'] = 1
        self.train_data.loc[(self.train_data['Fare'] > 14.454) & (self.train_data['Fare'] <= 31), 'Fare'] = 2
        self.train_data.loc[self.train_data['Fare'] > 31, 'Fare'] = 3
        self.train_data['Fare'] = self.train_data['Fare'].astype(int)

        self.test_data.loc[self.test_data['Fare'] <= 7.91, 'Fare'] = 0
        self.test_data.loc[(self.test_data['Fare'] > 7.91) & (self.test_data['Fare'] <= 14.454), 'Fare'] = 1
        self.test_data.loc[(self.test_data['Fare'] > 14.454) & (self.test_data['Fare'] <= 31), 'Fare'] = 2
        self.test_data.loc[self.test_data['Fare'] > 31, 'Fare'] = 3
        self.test_data['Fare'] = self.test_data['Fare'].astype(int)

    def naive_preprocessing_data(self):
        """
            This method is written just to have a data that will be classifiable to have a benchmark.
            We will do so :
            - Deletion of features that seems to be irrelevant
            - Converting a categorical feature to numerical feature
            - And completing missing value
        """

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
        y_test = self.gender_submission["Survived"]

        return x_train, y_train, x_test, y_test

    def advanced_preprocessing_data(self):
        # Here we delete features
        features_to_delete = ['PassengerId', 'Ticket', 'Cabin']

        self.train_data = self.delete_features(self.train_data, features_to_delete)
        self.test_data = self.delete_features(self.test_data, features_to_delete)

        # We affect the result to self.dataset
        self.dataset = [self.train_data, self.test_data]

        # Here we convert categorical features Sex to numerical
        self.train_data, self.test_data = self.change_categorical_feature_sex_to_numerical(self.train_data,
                                                                                           self.test_data)
        # Complete null values in the Age features
        self.completing_age_features()

        # Create title feature based on Name
        self.creating_title_feature()

        # Drop Name feature
        self.train_data = self.delete_features(self.train_data, ['Name'])
        self.test_data = self.delete_features(self.test_data, ['Name'])

        # Complete null values in the Embarked features
        self.completing_embarked_features()

        # Here we convert categorical features Embarked to numerical
        self.change_categorical_feature_embarked_to_numerical()

        # Create AgeBand feature
        self.create_age_band_features()

        # Create IsAlone feature
        self.create_is_alone_feature()

        # Complete null values in the Fare features
        self.completing_fare_features(self.test_data)

        # Create fare band
        self.create_fare_band()

        # delete some feature
        self.train_data = self.delete_features(self.train_data, ['Parch', 'SibSp', 'FamilySize'])
        self.test_data = self.delete_features(self.test_data, ['Parch', 'SibSp', 'FamilySize'])

        # Get the data
        x_train = self.delete_features(self.train_data, ['Survived'])
        y_train = self.train_data["Survived"]
        x_test = self.test_data
        y_test = self.gender_submission["Survived"]

        return x_train, y_train, x_test, y_test