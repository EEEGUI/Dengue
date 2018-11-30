from tqdm import tqdm
import lightgbm as lgb
import utils
import random
import itertools
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')


class ParamSearch:
    def __init__(self, train_array, train_label_array, param_grid, max_evals, **kwargs):
        self.train_set = lgb.Dataset(train_array, train_label_array)
        self.param_grid = param_grid
        self.max_evals = max_evals
        self.city = kwargs['city']

    def objective(self, hyperparameters, iteration):
        """Objective function for grid and random search. Returns
           the cross validation score from a set of hyperparameters."""

        # Number of estimators will be found using early stopping
        if 'n_estimators' in hyperparameters.keys():
            del hyperparameters['n_estimators']

            # Perform n_folds cross validation
        hyperparameters['verbose'] = -1
        hyperparameters['objective'] = 'regression'
        cv_results = lgb.cv(hyperparameters, self.train_set, nfold=3, num_boost_round=10000,
                            early_stopping_rounds=50, metrics='mae', shuffle=False)

        # results to retun
        score = cv_results['l1-mean'][-1]
        estimators = len(cv_results['l1-mean'])
        hyperparameters['n_estimators'] = estimators

        return [score, hyperparameters, iteration]

    def random_search(self):
        """Random search for hyperparameter optimization"""

        # Dataframe for results
        results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                               index=list(range(self.max_evals)))

        # Keep searching until reach max evaluations
        for i in tqdm(range(self.max_evals)):
            # Choose random hyperparameters
            hyperparameters = {k: random.sample(v, 1)[0] for k, v in self.param_grid.items()}
            hyperparameters['bagging_fraction'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                'bagging_fraction']

            # Evaluate randomly selected hyperparameters
            eval_results = self.objective(hyperparameters, i)
            print('score:%.5f:' % eval_results[0])
            results.loc[i, :] = eval_results

        # Sort with best score on top
        results.sort_values('score', ascending=True, inplace=True)
        results.reset_index(inplace=True)

        param_dict = results.loc[0, 'params']
        self._save_param(param_dict, '../config/%s_param_random.pkl' % self.city)
        print(param_dict)
        return results

    def grid_search(self):
        """Grid search algorithm (with limit on max evals)"""

        # Dataframe to store results
        results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                               index=list(range(self.max_evals)))

        keys, values = zip(*self.param_grid.items())

        i = 0

        # Iterate through every possible combination of hyperparameters
        for v in itertools.product(*values):

            # Create a hyperparameter dictionary
            hyperparameters = dict(zip(keys, v))

            # Set the subsample ratio accounting for boosting type
            hyperparameters['bagging_fraction'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                'bagging_fraction']

            # Evalute the hyperparameters
            eval_results = self.objective(hyperparameters, i)
            print('score:%.5f:' % eval_results[0])

            results.loc[i, :] = eval_results

            i += 1

            # Normally would not limit iterations
            if i > self.max_evals:
                break

        # Sort with best score on top
        results.sort_values('score', ascending=True, inplace=True)
        results.reset_index(inplace=True)

        param_dict = results.loc[0, 'params']
        self._save_param(param_dict, '../config/%s_param_grid.pkl' % self.city)
        print(param_dict)
        return results

    def _save_param(self, param_dict, param_path):
        with open(param_path, 'wb') as f:
            pickle.dump(param_dict, f)


if __name__ == '__main__':
    # df_train, df_train_label, _, __ = utils.load_data()
    # param_grid = {
    #     'boosting_type': ['gbdt', 'goss', 'dart'],
    #     'num_leaves': list(range(10, 150)),
    #     'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    #     'min_child_samples': list(range(5, 40)), # default 20
    #     'reg_alpha': list(np.linspace(0, 1)), # default 0 正则L1
    #     'reg_lambda': list(np.linspace(0, 1)), # default 0 正则L2
    #     'feature_fraction': list(np.linspace(0.6, 1, 10)), # 特征抽取 default 1.0
    #     'bagging_fraction': list(np.linspace(0.5, 1, 100)) # 数据抽取 default 1.0
    # }
    # param_search = ParamSearch(df_train, df_train_label, param_grid, 300)
    # print(param_search.random_search())

    data = utils.load_data_respectively()
    param_grid = {
        'boosting_type': ['gbdt', 'goss', 'dart'],
        'num_leaves': list(range(10, 150)),
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
        'min_child_samples': list(range(5, 40)), # default 20
        'reg_alpha': list(np.linspace(0, 1)), # default 0 正则L1
        'reg_lambda': list(np.linspace(0, 1)), # default 0 正则L2
        'feature_fraction': list(np.linspace(0.6, 1, 10)), # 特征抽取 default 1.0
        'bagging_fraction': list(np.linspace(0.5, 1, 100)) # 数据抽取 default 1.0
    }
    for each in data:
        param_search = ParamSearch(data[each][0], data[each][1], param_grid, 300, city=each)
        print(param_search.random_search())



