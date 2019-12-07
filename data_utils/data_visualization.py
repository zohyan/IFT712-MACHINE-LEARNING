import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from data_utils.data_loader import DataLoader


class DataVisualization:

    def __init__(self):
        self.path = 'datasets'
        self.train_data, self.test_data, gender_submission = DataLoader().load_data()

    def describe_data(self):
        print(self.train_data.describe())

    def get_data_columns(self):
        print(self.train_data.columns)

    def number_missing_values(self):
        column_names = self.train_data.columns
        for column in column_names:
            print('  ' + column + ' - ' + str(self.train_data[column].isnull().sum()))

    def target_visualization(self):
        dead_count = self.train_data.Survived.value_counts()[0]
        survived_count = self.train_data.Survived.value_counts()[1]
        dead = round((dead_count * 100) / (survived_count + dead_count), 2)
        survived = round((survived_count * 100) / (survived_count + dead_count), 2)
        labels = 'Mort', 'Survivant'
        sizes = [dead, survived]
        fig, ax = plt.subplots()
        ax.set_title('Balancement du dataset')
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        plt.show()

    def correlation_of_numerical_variable_with_the_target_class(self, feature):
        df = self.train_data[[feature, 'Survived']].groupby([feature], as_index=False).\
            mean().sort_values(by=feature, ascending=True)
        df.plot.bar(x=feature, y='Survived')
        plt.xticks(rotation=0)
        plt.ylim(0.0, 1.0)
        plt.title('Distribution de la probabilité de survie sachant la variable ' + "'" + feature + "'")
        plt.show()

    def correlation_of_age_variable_with_the_target_class(self):
        g = sns.FacetGrid(self.train_data, col='Survived', height=5)
        g.map(plt.hist, 'Age', bins=20)
        plt.show()

    def correlation_of_pclass_variable_with_the_target_class(self):
        grid = sns.FacetGrid(self.train_data, col='Survived', row='Pclass', height=2.2, aspect=1.6)
        grid.map(plt.hist, 'Age', alpha=.5, bins=20)
        grid.add_legend()
        plt.show()

    def correlation_of_categorical_variable_with_the_target_class(self):
        grid = sns.FacetGrid(self.train_data, row='Embarked', height=2.2, aspect=1.6)
        grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
        grid.add_legend()
        plt.show()

    def correlation_of_embarked_sex_variable_with_the_target_class(self):
        grid = sns.FacetGrid(self.train_data, row='Embarked', col='Survived', height=2.2, aspect=1.6)
        grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=None)
        grid.add_legend()
        plt.show()

dv = DataVisualization()
dv.correlation_of_embarked_sex_variable_with_the_target_class()
