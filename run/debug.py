import sys
import pandas as pd
import numpy as np
# import .model.experiment as ex
# import .module.data_processing as dpro
import gc
import argparse
import datetime
import subprocess
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

    return parser.parse_args()

if __name__ == "__main__":
    # args = get_args()
    # print(args.message)
    # slack_api = 'https://hooks.slack.com/services/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'
    with open('config/slack_api.json') as f:
        slack_api = json.load(f)['slack_api']

    fold_ = 1
    rmse = 10
    slack_text = f'FOLD{fold_} {rmse}'

    # cmd = "curl -X POST %s -d \"{'text': %s }\"" % (slack_api,slack_text)
    # print(cmd)

    text = ''' "''' + "{'text':"+"'"+str(slack_text)+"'"+"}" + '''" '''
    cmd = f"curl -X POST {slack_api} -d " + text
    subprocess.run(cmd,shell=True)