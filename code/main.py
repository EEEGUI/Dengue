import utils
from model import Model


SUBMISSION_PATH = '..\output\submission.csv'

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 15,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}


def main_v1():
    """
    两个城市一起训练
    :return:
    """
    df_train, df_train_label, df_test, df_test_pred = utils.load_data()
    # params = utils.load_param()

    model = Model(df_train, df_train_label, df_test, params)
    y_pred = model.train()

    utils.generate_submission(df_test_pred, y_pred, SUBMISSION_PATH)


def main_v2():
    """
    两个城市分开训练
    :return:
    """
    data = utils.load_data_respectively()
    prediction = []
    for each in data:
        # params = utils.load_param(each)
        model = Model(data[each][0], data[each][1], data[each][2], params)
        y_pred = model.train(city=each)
        prediction.append(utils.generate_submission(data[each][3], y_pred, None))
    df_submission = prediction[0].append(prediction[1])
    df_submission.to_csv(SUBMISSION_PATH, index=False)


if __name__ == '__main__':
    main_v2()



