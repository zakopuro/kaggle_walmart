from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
import gc

class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            setattr(self, f'lv{i + 1}_train_df', train_df.groupby(group_id)[train_target_columns].sum())
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        train_y = getattr(self, f'lv{lv}_train_df')
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = ((train_y.iloc[:, 1:].values - train_y.iloc[:, :-1].values) ** 2).mean(axis=1)
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        return np.mean(all_scores)


# class WRMSSEForLightGBM(WRMSSEEvaluator):

#     def feval(self, preds, dtrain):
#         if len(preds) > 853720:
#             preds = preds[-853720:]
#         preds = preds.reshape(self.valid_df[self.valid_target_columns].shape)
#         score = self.score(preds)
#         return 'WRMSSE', score, False


class WRMSSEForLightGBM(object):
    def __init__(self, product: pd.DataFrame,X_train: pd.DataFrame,sales_train_val: pd.DataFrame):
        NUM_ITEMS = 30490
        self.product = product
        weight_mat = np.c_[np.identity(NUM_ITEMS).astype(np.int8), #item :level 12
                   np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
                   pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values
                   ].T
        self.weight_mat_csr = csr_matrix(weight_mat)
        self.weight1, self.weight2 = self.weight_calc(X_train,product,sales_train_val)

    def weight_calc(self,data,product,sales_train_val):

        # calculate the denominator of RMSSE, and calculate the weight base on sales amount
        
        d_name = ['d_' + str(i+1) for i in range(1913)]
        
        sales_train_val = self.weight_mat_csr * sales_train_val[d_name].values
        
        # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
        # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
        df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(self.weight_mat_csr.shape[0],1)))
        
        start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
        
        
        # denominator of RMSSE / RMSSEの分母
        weight1 = np.sum((np.diff(sales_train_val,axis=1)**2),axis=1)/(1913-start_no)
        
        # calculate the sales amount for each item/level
        df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
        df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
        df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum).values
        
        weight2 = self.weight_mat_csr * df_tmp
        weight2 = weight2/np.sum(weight2)

        return weight1, weight2

    def feval(self, preds, dtrain):
        NUM_ITEMS = 30490
        # actual obserbed values / 正解ラベル
        y_true = dtrain.get_label()

        # number of columns
        num_col = len(y_true)//NUM_ITEMS
        
        # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
        reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
        reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
        
        # x_name = ['pred_' + str(i) for i in range(num_col)]
        # x_name2 = ["act_" + str(i) for i in range(num_col)]

        train = np.array(self.weight_mat_csr*np.c_[reshaped_preds, reshaped_true])
        
        score = np.sum(
                    np.sqrt(
                        np.mean(
                            np.square(
                                train[:,:num_col] - train[:,num_col:])
                            ,axis=1) / self.weight1) * self.weight2)
        
        return 'wrmsse', score, False