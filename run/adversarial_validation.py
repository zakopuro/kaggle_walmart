import sys
import pandas as pd
import numpy as np
sys.path.append('./')
import model.experiment as ex
from module import logger as logg
import module.data_processing as processing
import gc
import argparse
import subprocess
import datetime
import json
import pickle
from logging import Logger, Formatter, handlers, StreamHandler, getLogger
import warnings
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter('ignore')

features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90',
            'rolling_mean_t180', 'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30', 'rolling_skew_t30', 'rolling_kurt_t30']


def main():
    X_train = pd.read_pickle('data/other/validation.pkl')
    X_train = X_train.drop('demand',axis=1)
    X_test = pd.read_pickle('data/input/fe_test.pkl')
    X_train['target'] = 1
    X_test['target'] = 0

    dataset = pd.concat([X_train,X_test])
    X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset['target'],train_size=0.7)
    param = {'num_leaves': 50,
            'min_data_in_leaf': 30,
            'objective':'binary',
            'max_depth': 5,
            'learning_rate': 0.2,
            "min_child_samples": 20,
            "boosting": "gbdt",
            "feature_fraction": 0.9,
            "bagging_freq": 1,
            "bagging_fraction": 0.9 ,
            "bagging_seed": 44,
            "metric": 'auc',
            "verbosity": -1}
    train = lgb.Dataset(X_train, label=y_train)
    test = lgb.Dataset(X_test, label=y_test)
    num_round = 50
    clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.show()
    plt.savefig('adver-f-imp.png')


if __name__ == "__main__":
    main()
