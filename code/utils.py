import pandas as pd
from feature_engineer import Transformer2, Transformer
import pickle
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings('ignore')

TRAIN_FEATURE_PATH = r'../input/dengue_features_train.csv'
TRAIN_LABEL_PATH = r'../input/dengue_labels_train.csv'
TEST_FEATURE_PATH = '../input/dengue_features_test.csv'


def generate_submission(df_test, y_pred, path):
    df_test['total_cases'] = moving_average_2(y_pred)
    df_test['total_cases'] = df_test['total_cases'].apply(lambda x: int(x if x > 0 else 0))
    if not path:
        df_test.to_csv(path, index=False)
    return df_test


def generate_submission_2(sj_pred, iq_pred):
    df = pd.read_csv(TEST_FEATURE_PATH)
    df = df.loc[:, ['city', 'year', 'weekofyear']]
    y_pred = np.concatenate([sj_pred, iq_pred])
    df['total_cases'] = moving_average_2(y_pred)
    df['total_cases'] = df['total_cases'].apply(lambda x: int(x if x > 0 else 0))
    df.to_csv('../output/submission.csv', index=False)



def moving_average(y_pred, window_size=7):
    y_pred_head = y_pred[:window_size - 1]
    weights = np.exp(np.linspace(0, 1, window_size))
    weights = weights / np.sum(weights)
    y_pred_tail = np.convolve(weights, y_pred)[window_size - 1:-window_size + 1]
    return list(y_pred_head) + list(y_pred_tail)


def moving_average_2(y_pred, window_size=9):
    length = len(y_pred)
    weights = np.exp(np.linspace(0, 1, window_size//2 + 1))
    weights = np.concatenate([weights, weights[::-1][1:window_size//2]])
    weights = weights / np.sum(weights)
    for i in range(window_size//2, length-(window_size//2)):
        y_pred[i] = np.sum(y_pred[i - window_size//2: i+window_size//2] * weights)
    return y_pred


def load_data():
    """
    加载数据，并做预处理
    返回训练集特征，训练集标签， 测试集特征， 测试集用于提交的列
    :return:
    """
    df_train = pd.read_csv(TRAIN_FEATURE_PATH)
    df_train_label = pd.read_csv(TRAIN_LABEL_PATH)
    df_test = pd.read_csv(TEST_FEATURE_PATH)

    transform = Transformer(df_train, True, '../config/label_enconding.pkl')
    df_train = transform().values

    df_test_pred = df_test.loc[:, ['city', 'year', 'weekofyear']]
    transform = Transformer(df_test, False, '../config/label_enconding.pkl')
    df_test = transform().values

    df_train_label = df_train_label.loc[:, 'total_cases'].values
    return df_train, df_train_label, df_test, df_test_pred


def load_data_by_city(city):
    df_train_feature = pd.read_csv(TRAIN_FEATURE_PATH)
    df_train_label = pd.read_csv(TRAIN_LABEL_PATH)
    df_test_feature = pd.read_csv(TEST_FEATURE_PATH)

    train_feature = df_train_feature[df_train_feature['city'] == city]
    train_label = df_train_label[df_train_label['city'] == city]
    test_feature = df_test_feature[df_test_feature['city'] == city]

    transform = Transformer2(train_feature, True, city,
                             no_process=[])  # 'city', 'year', 'weekofyear', 'week_start_date'
    train_feature = transform()

    submission_cols = test_feature.loc[:, ['city', 'year', 'weekofyear']]
    transform = Transformer2(test_feature, False, city)
    test_feature = transform()

    train_label = train_label.loc[:, 'total_cases'].values

    return {'train_feature': train_feature,
            'train_label': train_label,
            'test_feature': test_feature,
            'submission_cols': submission_cols}


def load_data_respectively():
    """
    分城市加载数据
    :return:[[sj 训练集特征、训练集标签、测试集特征、测试集前几列（用于生成submission）],[iq]]
    """
    df_train_feature = pd.read_csv(TRAIN_FEATURE_PATH)
    df_train_label = pd.read_csv(TRAIN_LABEL_PATH)
    df_test_feature = pd.read_csv(TEST_FEATURE_PATH)

    result = {}
    for city in ['sj', 'iq']:
        train_feature = df_train_feature[df_train_feature['city'] == city]
        train_label = df_train_label[df_train_label['city'] == city]
        test_feature = df_test_feature[df_test_feature['city'] == city]

        transform = Transformer2(train_feature, True, city, no_process=[])   # 'city', 'year', 'weekofyear', 'week_start_date'
        train_feature = transform().values

        submission_cols = test_feature.loc[:, ['city', 'year', 'weekofyear']]
        transform = Transformer2(test_feature, False, city)
        test_feature = transform().values

        train_label = train_label.loc[:, 'total_cases'].values
        result[city] = [train_feature, train_label, test_feature, submission_cols]

    return result


def load_param(city):
    """
    加载超参数
    :return:
    """
    with open('../config/%s_param_random.pkl' % city, 'rb') as f:
        param = pickle.load(f)
    return param


def plot_fit_situation(label, pred, name, show=True):
    """
    数据拟合情况可视化
    :param label:
    :param pred:
    :param name:
    :param show:
    :return:
    """
    fig = plt.figure(figsize=(30, 10))
    ax = fig.subplots()
    ax.plot(label)
    ax.plot(pred)
    ax.plot(moving_average_2(pred))

    ax.set_title('Total cases fitting situation'), ax.set_ylabel('Total cases'), ax.legend(['label', 'pred', 'mog_pred'])
    plt.savefig('../output/%s_train.png' % name)
    if show:
        plt.show()


def plot_result():
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    data = pd.read_csv('../output/submission.csv')
    data['total_cases'].plot(ax=ax)
    ax.set_title('Total cases fitting situation'), ax.set_ylabel('Total cases'), ax.legend(['pred'])
    plt.show()


def generate_sequence_feature(features, sequence_length):
    features = pd.DataFrame(features)
    cols_name = features.columns
    for i in range(1, sequence_length):
        for col_name in cols_name:
            features['%s_%d' % (col_name, i)] = 0

    for i in range(1, sequence_length):
        for col_name in cols_name:
            if i == 1:
                features.loc[i:, '%s_%d' % (col_name, i)] = features.loc[i:, col_name]
            else:
                features.loc[i:, '%s_%d' % (col_name, i)] = features.loc[i:, '%s_%d' % (col_name, i - 1)]
    return features.values


if __name__ == '__main__':
    # load_data_respectively()

    # plot_result()
    data = load_data_by_city('sj')
    df = generate_sequence_feature(data['train_feature'], 52)
    print(len(df.columns))