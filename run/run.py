import sys
import pandas as pd
import numpy as np
sys.path.append('./')
import model.experiment as ex
from module import logger as logg
import module.data_processing as processing
from module.walmart_metric import WRMSSEEvaluator,WRMSSEForLightGBM
import gc
import argparse
import subprocess
import datetime
import json
import pickle
from logging import Logger, Formatter, handlers, StreamHandler, getLogger
import warnings
warnings.simplefilter('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    now_time = datetime.datetime.now()
    now_time_str = f'{str(now_time.month)}_{str(now_time.day)}_{str(now_time.hour)}_{str(now_time.minute)}_{str(now_time.second)}'
    # 必須


    # 任意
    parser.add_argument('-m',"--message", type=str,default=str(now_time_str))
    parser.add_argument('-f',"--file_name", type=str,default=str(now_time_str))
    parser.add_argument("-s","--sub",default=False,action="store_true")

    del now_time
    gc.collect()

    return parser.parse_args()

def load_datasets(feats):
    dfs = [pd.read_pickle(f'features/{f}_train.pkl') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_pickle(f'features/{f}_test.pkl') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test

def main(args,logger):
    logger.info('START')
    train_df = pd.read_csv('data/other/sales_train_validation.csv.zip')
    train_fold_df = train_df.iloc[:, :-28]
    valid_fold_df = train_df.iloc[:, -28:]
    calendar = pd.read_csv('data/other/calendar.csv')
    prices = pd.read_csv('data/other/sell_prices.csv.zip')
    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
    lgb_evaluator = WRMSSEForLightGBM(train_fold_df, valid_fold_df, calendar, prices)
    del train_df,train_fold_df,valid_fold_df,calendar,prices
    gc.collect()

    with open('config/config.json') as f:
        cfg = json.load(f)
        features = cfg['features']
        lgb_params = cfg['lgb_params']
        feats = cfg['FE']

    file_name = args.file_name
    with open('config/slack_api.json') as f:
        slack_api = json.load(f)['slack_api']
    X_train,X_test = load_datasets(feats)
    y_train = X_train['demand']

    logger.info('Loaded data')

    # 学習/予測
    lgb_models,lgb_oof,lgb_predictions = ex.run_lightgbm(X_train=X_train,y_train=y_train,X_test=X_test,early_stopping_round=50,splits=1,features= features,
                                                        file_name=file_name,slack_api=slack_api,logger=logger,metric=evaluator,lgb_metric=lgb_evaluator,
                                                        params=lgb_params)

    del X_train,y_train
    gc.collect()

    # 後処理
    X_test['demand'] = lgb_predictions
    # 一時ファイル
    X_test[['demand']].to_pickle(f'data/submit/{file_name}_submission.pkl')
    sample_sub = pd.read_csv('data/submit/sample_submission.csv.zip')

    sub = processing.postprocessing(X_test,sample_sub)

    del X_test
    gc.collect()

    # sumit
    sub.to_csv(f'data/submit/{file_name}_submission.csv.zip',compression='zip',index=False)

    # save model
    with open(f'model/models/{file_name}_model.pkl','wb') as f:
        pickle.dump(lgb_models,f)

    # save oof
    with open(f'data/oof/{file_name}_oof.pkl','wb') as f:
        pickle.dump(lgb_oof,f)


    if args.sub:
        message = args.message
        submit_cmd = f'kaggle competitions submit -c m5-forecasting-accuracy -f data/submit/{file_name}_submission.csv.zip -m "{message}"'
        subprocess.run(submit_cmd,shell=True)
        logger.info('kaggle submit')

        slack_text = 'submit完了'
        text = ''' "''' + "{'text':"+"'"+str(slack_text)+"'"+"}" + '''" '''
        cmd = f"curl -X POST {slack_api} -d " + text
        subprocess.run(cmd,shell=True)

    logger.info('END')

if __name__ == "__main__":
    args = get_args()
    logger = logg.Logger(name='walmart',filename=args.file_name)
    # try:
    main(args,logger)
    # except:
    #     with open('config/slack_api.json') as f:
    #         slack_api = json.load(f)['slack_api']
    #     slack_text = "error"
    #     text = ''' "''' + "{'text':"+"'"+str(slack_text)+"'"+"}" + '''" '''
    #     cmd = f"curl -X POST {slack_api} -d " + text
    #     subprocess.run(cmd,shell=True)

    #     logger.error('error')