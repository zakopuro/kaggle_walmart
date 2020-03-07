import sys
import pandas as pd
import numpy as np
import model.experiment as ex
import module.data_processing as dpro
import gc
import argparse
import subprocess
import datetime
import json

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



def main():
    args = get_args()
    X_train = pd.read_pickle('data/')
    y_train = pd.read_pickle('data/')
    X_test = pd.read_pickle('data/')
    file_name = args.file_name
    with open('config/slack_api.json') as f:
        slack_api = json.load(f)['slack_api']

    # 学習/予測
    lgb_models,lgb_oof,lgb_predictions = ex.run_lightgbm(X_train=X_train,y_train=y_train,X_test=X_test,file_name=file_name)


    # 後処理
    tmp_sub = X_test[['id']]
    tmp_sub['pred'] = lgb_predictions
    del X_train,y_train,X_test
    gc.collect()
    tmp_sub = dpro.postprocessing(tmp_sub)
    sample_submission = pd.read_csv('data/submit/sample_submission.csv.zip')
    sample_submission.iloc[:len(tmp_sub)] = tmp_sub

    # sumit
    sample_submission.to_csv(f'data/submit/{file_name}_submission.csv.zip',compression='zip',index=False)

    if args.sub:
        message = args.message
        submit_cmd = f'kaggle competitions submit -c m5-forecasting-accuracy -f ../data/sumit/{file_name}_submission.csv.zip -m "{message}"'
        subprocess.run(submit_cmd,shell=True)



if __name__ == "__main__":
    main()