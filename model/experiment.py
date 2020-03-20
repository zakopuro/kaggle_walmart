import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('./')
from module.walmart_metric import WRMSSEEvaluator,WRMSSEForLightGBM
from logging import Logger, getLogger
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import subprocess

features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'dayofweek', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90',
            'rolling_mean_t180', 'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30', 'rolling_skew_t30', 'rolling_kurt_t30']


def wrmsse_oof(oof):
    df_oof = pd.DataFrame()
    for i,d in enumerate(range(1886,1914,1)):
        df_oof['d_'+str(d)] = oof[i*30490:(i+1)*30490]
    return df_oof


def run_lightgbm(X_train : pd.DataFrame, y_train : pd.DataFrame, X_test : pd.DataFrame,
                 file_name : str = None, logger = None, cat_feature : list = None,
                 with_mlflow : bool = False, with_optuna : bool = False, metric : WRMSSEEvaluator = None,lgb_metric : WRMSSEForLightGBM = None,
                 splits : int = 5, params : dict = None,slack_api = None,
                 early_stopping_round: int=None, save_feature_imp : bool = False):

    logger.info('start lgb train')
    if params == None:
        params = {
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'objective': 'regression',
            'n_jobs': -1,
            'seed': 236,
            'learning_rate': 0.1,
            'bagging_fraction': 0.75,
            'bagging_freq': 10,
            'colsample_bytree': 0.75}
    if not early_stopping_round is None:
        params['early_stopping_rounds'] = early_stopping_round

    if with_optuna:
        # import optuna.integration.lightgbm as lgb
        a = 1

    if with_mlflow:
        from mlflow.lightgbm import autolog
        autolog()

    oof = np.zeros(len(X_train))
    oof_idx = []
    predictions = np.zeros(len(X_test))
    feature_imp = pd.DataFrame()
    models = []
    val_date_list = []
    X_train['date'] = pd.to_datetime(X_train['date'])
    # X_test['date'] = pd.to_datetime(X_test['date'])

    max_date = X_train['date'].max()
    for fold_ in range(splits):
        logger.info(f"Fold {fold_}")
        if fold_ == 0:
            val_date = max_date - np.timedelta64(28,'D')
            val_date_list.append(val_date)
            valid_x = X_train[(X_train['date']>val_date)][features]
            train_x = X_train[(X_train['date']<=val_date)][features]

        else:
            val_date = val_date - np.timedelta64(28,'D')
            val_date_list.append(val_date)
            valid_x = X_train[(X_train['date']>val_date_list[fold_]) & (X_train['date']<=val_date_list[fold_-1])][features]
            train_x = X_train[(X_train['date']<=val_date)][features]

        valid_y = y_train.iloc[valid_x.index]
        train_y = y_train.iloc[train_x.index]

        trn_data = lgb.Dataset(train_x, label=train_y)
        val_data = lgb.Dataset(valid_x, label=valid_y)

        lgb_model = lgb.train(params,
                        trn_data,
                        num_boost_round = 2500,
                        valid_sets = [trn_data, val_data],
                        # feval = lgb_metric.feval,
                        verbose_eval=100)

        pred = lgb_model.predict(valid_x)
        idx = list(valid_x.index)
        oof[idx] = pred
        oof_idx += idx
        predictions += lgb_model.predict(X_test[features]) / splits
        models.append(lgb_model)
        mse = mean_squared_error(pred, valid_y)
        rmse = np.sqrt(mse)
        logger.info(f"rmse: {rmse}")

        feature_imp = pd.DataFrame()
        feature_imp['fold_'+str(fold_)] = lgb_model.feature_importance()
        del lgb_model, train_x, train_y, valid_x, valid_y
        gc.collect()

        slack_text = f'FOLD{fold_} {rmse}'
        text = ''' "''' + "{'text':"+"'"+str(slack_text)+"'"+"}" + '''" '''
        cmd = f"curl -X POST {slack_api} -d " + text
        subprocess.run(cmd,shell=True)
    # feature_imp.index = X_train.columns
    oof = oof[oof_idx]
    oof_mse = mean_squared_error(oof, y_train.iloc[oof_idx].values)
    oof_rmse = np.sqrt(oof_mse)
    logger.info(f"oof rmse: {oof_rmse}")
    df_oof = wrmsse_oof(oof)
    oof_wrmsse = metric.score(df_oof)
    logger.info(f"oof wrmsse: {oof_wrmsse}")

    slack_text = f'OOF RMSE {oof_rmse}'
    text = ''' "''' + "{'text':"+"'"+str(slack_text)+"'"+"}" + '''" '''
    cmd = f"curl -X POST {slack_api} -d " + text
    subprocess.run(cmd,shell=True)

    slack_text = f'OOF WRMSSE {oof_wrmsse}'
    text = ''' "''' + "{'text':"+"'"+str(slack_text)+"'"+"}" + '''" '''
    cmd = f"curl -X POST {slack_api} -d " + text
    subprocess.run(cmd,shell=True)


    if save_feature_imp:
        feature_imp['all_fold'] = feature_imp.mean(axis=1)
        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig(f'features/feature_imp/lgbm_importances_{file_name}.png')

        logger.info('save feature imp')

    logger.info('end lgb train')
    return models,oof,predictions


# TODO
# xgb

# cat

# NN