import pandas as pd
import numpy as np
import gc
import os
from base_create_features import Feature, get_arguments, generate_features
from pathlib import Path
Feature.dir = 'features'

def _label_encoder(data):
    l_data,_ =data.factorize(sort=True)
    if l_data.max()>32000:
        l_data = l_data.astype('int32')
    else:
        l_data = l_data.astype('int16')

    if data.isnull().sum() > 0:
        l_data = np.where(l_data == -1,np.nan,l_data)
    return l_data

def _reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def _read_data():
    # use_col = ['id','item_id','dept_id','cat_id','store_id','state_id']
    # d_col = ['d_'+str(d) for d in range(1913,1913-days,-1)]
    # use_col += d_col
    print('Reading files...')
    calendar = pd.read_csv('data/other/calendar.csv')
    calendar = _reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv('data/other/sell_prices.csv.zip')
    sell_prices = _reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv('data/other/sales_train_validation.csv.zip')
    # sales_train_validation = sales_train_validation[use_col]
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv('data/submit/sample_submission.csv.zip')
    return calendar, sell_prices, sales_train_validation, submission

def _melt_and_merge(calendar, sell_prices, sales_train_validation, submission, days = 365, merge = False):

    # melt sales data, get it ready for training
    sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    sales_train_validation = _reduce_mem_usage(sales_train_validation)

    # seperate test dataframes
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    test2 = submission[submission['id'].isin(test2_rows)]

    # change column names
    test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
                      'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
    test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
                      'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']

    # get product table
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    # merge with product table
    test2['id'] = test2['id'].str.replace('_evaluation','_validation')
    test1 = test1.merge(product, how = 'left', on = 'id')
    test2 = test2.merge(product, how = 'left', on = 'id')
    test2['id'] = test2['id'].str.replace('_validation','_evaluation')

    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    sales_train_validation['part'] = 'train'
    nrows = sales_train_validation['id'].nunique()*(1913-(days+28))
    sales_train_validation[nrows:]
    test1['part'] = 'test1'
    test2['part'] = 'test2'

    data = pd.concat([sales_train_validation, test1, test2], axis = 0)

    del sales_train_validation, test1, test2

    # get only a sample for fst training
    # data = data.loc[nrows:]

    # drop some calendar features
    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)

    # delete test2 for now
    data = data[data['part'] != 'test2']

    if merge:
        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
        data.drop(['d', 'day'], inplace = True, axis = 1)
        # get the sell price data (this feature should be very important)
        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
        data = data[~data['sell_price'].isnull()].reset_index(drop=True)
        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    else:
        pass

    gc.collect()

    return data


class Date(Feature):
    def create_features(self):
        data = pd.concat([train,test])

        data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
        data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
        data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
        data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
        data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
        data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
        data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
        data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
        data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
        data['rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
        data['rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())

        self.train = data[data['part'] == 'train'].drop(drop_col,axis=1)
        self.test = data[data['part'] == 'test1'].drop(drop_col,axis=1)

class Price(Feature):
    def create_features(self):
        data = pd.concat([train,test])
        data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
        data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
        data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
        data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
        data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
        data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
        data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
        self.train = data[data['part'] == 'train'].drop(drop_col,axis=1)
        self.test = data[data['part'] == 'test1'].drop(drop_col,axis=1)

class Time(Feature):
    def create_features(self):
        data = pd.concat([train,test])
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['week'] = data['date'].dt.week
        data['day'] = data['date'].dt.day
        data['dayofweek'] = data['date'].dt.dayofweek

        self.train = data[data['part'] == 'train'].drop(drop_col,axis=1)
        self.test = data[data['part'] == 'test1'].drop(drop_col,axis=1)


class Weather(Feature):
    def create_features(self):
        data = pd.concat([train,test])
        data = pd.merge(data,weather,on=['state_id','date'],how='left')
        # LE
        data['AM_weather'] = data['AM_weather'].replace({'sunny':0,'cloudy':1,'rain':2,'snow':3})
        data['PM_weather'] = data['PM_weather'].replace({'sunny':0,'cloudy':1,'rain':2,'snow':3})

        self.train = data[data['part'] == 'train'].drop(drop_col,axis=1)
        self.test = data[data['part'] == 'test1'].drop(drop_col,axis=1)




if __name__ == "__main__":
    args = get_arguments()
    Base_train_path = 'features/Base_train.pkl'
    Base_test_path = 'features/Base_test.pkl'
    if os.path.exists(Base_train_path) and os.path.exists(Base_test_path) and not args.force:
        train = pd.read_pickle(Base_train_path)
        test = pd.read_pickle(Base_test_path)
        print('Base skipped')
    else:
        calendar, sell_prices, sales_train_validation, submission = _read_data()
        data = _melt_and_merge(calendar, sell_prices, sales_train_validation, submission,merge = True,days=args.days)
        not_le_col = ['part','date','id']
        for col in data.columns:
            if (data[col].dtype == object) and (not col in not_le_col):
                data[col] = _label_encoder(data[col])
        train = data[data['part'] == 'train']
        test = data[data['part'] == 'test1']
        train.to_pickle(Base_train_path)
        test.to_pickle(Base_test_path)

    drop_col = test.columns
    weather = pd.read_pickle('data/other/weather.pkl')

    generate_features(globals(),args.force)