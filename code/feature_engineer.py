import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Imputer
import pickle


class Transformer:
    """
    数据变换类
    把对训练集的特征工程应用到测试集
    """
    def __init__(self, df, is_train, config_path='../config/label_enconding.pkl'):
        self.df = df
        self.original_cols = df.columns
        self.is_train = is_train
        self.config_path = config_path
        if self.is_train:
            self.config = {}
        else:
            if config_path is None:
                raise ValueError('请加载配置文件！')
            else:
                with open(config_path, 'rb') as f:
                    self.config = pickle.load(f)

    def _to_datetime(self):
        # string -> datetime
        self.df['week_start_date'] = pd.to_datetime(self.df.week_start_date, format='%Y/%m/%d')
        # 把周开始的时间转换成月份
        self.df['week_start_date'] = self.df['week_start_date'].apply(lambda x: x.month)

    def _label_enconding(self):
        # 两种类别的编码
        # 处理训练集的时候，记录编码对应的类别
        if self.is_train:
            le = LabelEncoder()
            le_config = {}
            for col in self.df:
                if self.df[col].dtype == 'object':
                    le.fit(self.df[col])
                    self.df[col] = le.transform(self.df[col])
                    le_config[col] = le
            self.config['label_enconding'] = le_config

        # 处理测试集时，利用保存的编码方式对测试集进行编码
        else:
            le_config = self.config['label_enconding']
            for col in le_config:
                self.df[col] = le_config[col].transform(self.df[col])

    def __call__(self):
        self._to_datetime()
        self._label_enconding()
        if self.is_train:
            with open(self.config_path, 'wb') as f:
                pickle.dump(self.config, f)
        return self.df


class Transformer2:
    """
    数据变换类
    把对训练集的特征工程应用到测试集
    不进行城市类别编码
    """

    def __init__(self, df, is_train, city, **kwargs):
        self.df = df
        self.original_cols = df.columns
        self.is_train = is_train
        self.city = city
        self.other_params = kwargs
        self.ef_file = '..\config\%s_min_max_scale.pkl' % self.city

        if self.is_train:
            self.config = {}
        else:
            with open(self.ef_file, 'rb') as f:
                self.config = pickle.load(f)

    def _to_datetime(self):
        # string -> datetime
        self.df['week_start_date'] = pd.to_datetime(self.df.week_start_date, format='%Y/%m/%d')
        # 把周开始的时间转换成月份
        self.df['week_start_date'] = self.df['week_start_date'].apply(lambda x: x.month)

    def _drop_cols(self, drop_cols):
        self.df = self.df.drop(labels=drop_cols, axis=1)

    def _min_max_scale(self):
        # 处理训练集的时候，归一化
        if self.is_train:
            scale = MinMaxScaler()
            cols_to_process = list(set(self.df.columns) - set(self.other_params['no_process']))

            scale.fit(self.df.loc[:, cols_to_process])
            self.df.loc[:, cols_to_process] = scale.transform(self.df.loc[:, cols_to_process])

            self.config['min_max_scale'] = {'cols_to_process': cols_to_process, 'scaler': scale}
            with open(self.ef_file, 'wb') as f:
                pickle.dump(self.config, f)

        else:
            scale_config = self.config['min_max_scale']
            cols_to_process = scale_config['cols_to_process']
            scale = scale_config['scaler']
            self.df.loc[:, cols_to_process] = scale.transform(self.df.loc[:, cols_to_process])

    def _fill_nan(self):
        imputer = Imputer(strategy='median')
        imputer.fit(self.df)
        self.df = imputer.transform(self.df)

    def __call__(self):
        self._to_datetime()
        self._drop_cols(['city', 'reanalysis_sat_precip_amt_mm'])
        self._min_max_scale()
        self._fill_nan()
        return self.df


if __name__ == '__main__':
    df_train = pd.read_csv(r'..\input\dengue_features_train.csv')
    # transform = Transformer(df_train, True, '../config/label_enconding.pkl')
    # df_train = transform()
    # print(df_train.head())
    df_test = pd.read_csv('../input/dengue_features_test.csv')
    transform = Transformer(df_test, False, '../config/label_enconding.pkl')
    df_test = transform()
    print(df_test.head())

