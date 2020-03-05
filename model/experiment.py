import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn
import mlflow
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from logging import Logger, getLogger
from mlflow.lightgbm import autolog
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def run_lightgbm(X_train:pd.DataFrame,y_train:pd.DataFrame,X_test:pd.DataFrame,file_name : str = None,
                 cat_feature : list = None, with_log : bool = False,
                 with_mlflow : bool = False, with_optuna : bool = False,
                 splits : int = 5, params : dict = None,
                 early_stopping_round: int=None, save_feature_imp : bool = False):

    if not file_name is None:
        now = datetime.datetime.now
        file_name = now

    if params == None:
        params = {
                'boosting_type': 'gbdt',
                'metric': 'rmse',
                'objective': 'regression',
                'n_jobs': -1,
                'seed': 127,
                'learning_rate': 0.1,
                'bagging_fraction': 0.75,
                'bagging_freq': 10,
                'colsample_bytree': 0.75
                }
    if not early_stopping_round is None:
        params['early_stopping_rounds'] = early_stopping_round

    if with_optuna:
        import optuna.integration.lightgbm as lgb

    if with_mlflow:
        autolog()

    if with_log:
        logger = getLogger('lgb_train')

    folds = KFold(n_splits=splits)
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_test))
    feature_imp = pd.DataFrame()
    models = []
    for fold_,(trn_idx,val_idx) in enumerate(folds.split(X_train.values,y_train.values)):
        logger.info(f"Fold {fold_}")
        train_x, train_y = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        valid_x, valid_y = X_train.iloc[val_idx], y_train.iloc[val_idx]

        trn_data = lgb.Dataset(train_x, label=train_y)
        val_data = lgb.Dataset(valid_x, label=valid_y)

        lgb_model = lgb.train(params,
                        trn_data,
                        10000,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500)

        pred = lgb_model.predict(valid_x)
        oof[val_idx] = pred
        predictions += lgb_model.predict(X_test) / splits

        models.append(lgb_model)

        mse = mean_squared_error(pred, valid_y)
        rmse = np.sqrt(mse)
        logger.info(f"rmse: {rmse}")

        feature_imp['fold_'+str(fold_)] = pd.DataFrame(sorted(zip(lgb_model.feature_importances_,X_train.columns)), columns=['Value','Feature'])
        del lgb_model, train_x, train_y, valid_x, valid_y
        gc.collect()

    oof_mse = mean_squared_error(predictions, y_train)
    oof_rmse = np.sqrt(oof_mse)
    logger.info(f"oof rmse: {oof_rmse}")

    if save_feature_imp:
        feature_imp['all_fold'] = feature_imp.mean(axis=1)
        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig(f'lgbm_importances_{file_name}.png')


    return models,oof,predictions