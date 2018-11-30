import lightgbm as lgb
from sklearn.model_selection import train_test_split
import utils


MODEL_SAVE_PATH = r'C:\Users\14622\PycharmProjects\Dengue\config\model.pkl'


class Model:
    def __init__(self, train_features, train_labels, test_features, params):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.params = params

    def train(self, **kwargs):
        EVAL_SIZE = 0.3
        train_x, val_x, train_y, val_y = train_test_split(self.train_features,
                                                          self.train_labels,
                                                          test_size=EVAL_SIZE,
                                                          shuffle=False)
        lgb_trian = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(val_x, val_y)
        gbm = lgb.train(self.params, lgb_trian, num_boost_round=10000, valid_sets=lgb_eval, early_stopping_rounds=100)

        print('Saving model...')
        # save model to file
        gbm.save_model(MODEL_SAVE_PATH)

        print('Starting predicting...')
        # predict
        y_train_pred = gbm.predict(self.train_features, num_iteration=gbm.best_iteration)
        y_pred = gbm.predict(self.test_features, num_iteration=gbm.best_iteration)
        utils.plot_fit_situation(self.train_labels, y_train_pred, kwargs['city'], show=False)
        print('finish!')
        return y_pred
