import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


class DataVisualization:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.num_cols = len(df.columns)
        sns.set(style="darkgrid")

    def plot_missing_matrix(self):
        msno.matrix(self.df)
        plt.tight_layout()
        plt.show()

    def plot_missing_bar(self):
        msno.bar(self.df)
        plt.tight_layout()
        plt.show()

    def plot_data_distribution(self):
        num_col = int((self.num_cols * 2)**0.5) + 1
        num_row = self.num_cols // num_col + 1
        f, axes = plt.subplots(num_row, num_col, figsize=(19, 10))

        for ax, col in zip(axes.flat, self.df.columns):
            if self.df[col].dtype == 'object':
                sns.countplot(x=col, data=self.df, ax=ax)
            elif self.df[col].dtype in ['int64', 'float64']:
                sns.distplot(self.df[col], hist=False, ax=ax)
            else:
                pass

        plt.tight_layout()
        plt.show()

    def plot_data_relation(self):
        f, ax = plt.subplots(figsize=(19, 10))
        corr = self.df.corr()
        sns.heatmap(corr, cmap="RdBu_r", annot=True, ax=ax)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../input/dengue_features_train.csv')
    df_label = pd.read_csv('../input/dengue_labels_train.csv')
    df['total_cases'] = df_label['total_cases']
    data = DataVisualization(df, 'total_cases')
    # data.plot_data_distribution()
    # data.plot_missing_bar()
    data.plot_data_relation()